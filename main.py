import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RBFInterpolator, griddata
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from pykrige.ok import OrdinaryKriging


BASE = Path.cwd()
DATA = BASE / "data" 
OUT = BASE / "project_outputs"
FIG = OUT / "figures"
GRID_DIR = OUT / "gridded_daily_2deg"

STATION_FILE = DATA / "All_CONUS_Station_Information_V2.csv"
VARIABLE_FILES = {
    "Air temperature": DATA / "USCRN_AirTemperature_2006_2021.csv",
    "Precipitation": DATA / "USCRN_Precipitation_2006_2021.csv",
    "Relative humidity": DATA / "USCRN_RelativeHumidity_2006_2021.csv",
    "Soil moisture 10cm": DATA / "USCRN_SoilMoisture10cm_2006_2021.csv",
    "Soil temperature": DATA / "USCRN_SoilTemperature_2006_2021.csv",
}

ERA5_YEAR = "2015"
ERA5_FILES = {
    "t2m": DATA / f"era5_t2m_{ERA5_YEAR}.nc",
    "tp": DATA / f"era5_tp_{ERA5_YEAR}.nc",
    "d2m": DATA / f"era5_d2m_{ERA5_YEAR}.nc",
    "stl1": DATA / f"era5_stl1_{ERA5_YEAR}.nc",
    "swvl1": DATA / f"era5_swvl1_{ERA5_YEAR}.nc",
}


def ensure_dirs():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    GRID_DIR.mkdir(exist_ok=True)


def safe_nanmean(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan


def load_station_info():
    stations = pd.read_csv(STATION_FILE)
    stations.columns = stations.columns.str.strip()
    stations = stations.drop_duplicates("StationName")
    stations = stations.rename(columns={"Lon": "lon", "Lat": "lat"})
    return stations.query("lon >= -125 and lon <= -66 and lat >= 24 and lat <= 50").copy()


def load_variable(path, station_names):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cols = ["date"] + [c for c in station_names if c in df.columns]
    return df[cols]


def make_grid():
    lons = np.arange(-125, -65.99, 2.0)
    lats = np.arange(24, 50.01, 2.0)
    lon2d, lat2d = np.meshgrid(lons, lats)
    grid = pd.DataFrame({"lon": lon2d.ravel(), "lat": lat2d.ravel()})
    return lons, lats, grid


def idw_predict(train_xy, train_z, pred_xy, power=2, k=12):
    mask = np.isfinite(train_z)
    train_xy = train_xy[mask]
    train_z = train_z[mask]
    if len(train_z) == 0:
        return np.full(len(pred_xy), np.nan)

    k = min(k, len(train_z))
    tree = cKDTree(train_xy)
    dist, idx = tree.query(pred_xy, k=k)

    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    if dist.shape[0] != len(pred_xy):
        dist = dist.T
        idx = idx.T

    exact = dist == 0
    weights = 1.0 / np.maximum(dist, 1e-12) ** power
    pred = np.sum(weights * train_z[idx], axis=1) / np.sum(weights, axis=1)

    if exact.any():
        rows = np.where(exact.any(axis=1))[0]
        for r in rows:
            pred[r] = train_z[idx[r, np.argmax(exact[r])]]
    return pred


def linear_predict(train_xy, train_z, pred_xy):
    mask = np.isfinite(train_z)
    train_xy = train_xy[mask]
    train_z = train_z[mask]
    if len(train_z) < 4:
        return np.full(len(pred_xy), np.nan)
    pred = griddata(train_xy, train_z, pred_xy, method="linear")
    missing = ~np.isfinite(pred)
    if missing.any():
        pred[missing] = griddata(train_xy, train_z, pred_xy[missing], method="nearest")
    return pred


def rbf_predict(train_xy, train_z, pred_xy, kernel):
    mask = np.isfinite(train_z)
    train_xy = train_xy[mask]
    train_z = train_z[mask]
    if len(train_z) < 8:
        return np.full(len(pred_xy), np.nan)
    model = RBFInterpolator(train_xy, train_z, kernel=kernel, neighbors=min(40, len(train_z)))
    return model(pred_xy)


def kriging_predict(train_xy, train_z, pred_xy, variogram_model="spherical"):
    mask = np.isfinite(train_z)
    train_xy = train_xy[mask]
    train_z = train_z[mask]

    if len(train_z) < 10:
        return np.full(len(pred_xy), np.nan)

    try:
        ok = OrdinaryKriging(
            train_xy[:, 0],
            train_xy[:, 1],
            train_z,
            variogram_model=variogram_model,
            verbose=False,
            enable_plotting=False,
        )
        z, _ = ok.execute("points", pred_xy[:, 0], pred_xy[:, 1])
        return np.asarray(z, dtype=float)
    except Exception:
        return np.full(len(pred_xy), np.nan)


MODEL_FUNCS = {
    "IDW": lambda x, z, p: idw_predict(x, z, p),
    "Linear": linear_predict,
    "RBF linear": lambda x, z, p: rbf_predict(x, z, p, "linear"),
    "RBF thin_plate_spline": lambda x, z, p: rbf_predict(x, z, p, "thin_plate_spline"),
    "Kriging spherical": lambda x, z, p: kriging_predict(x, z, p, "spherical"),
    "Kriging circular": lambda x, z, p: kriging_predict(x, z, p, "circular"),
    "Kriging gaussian": lambda x, z, p: kriging_predict(x, z, p, "gaussian"),
}


def pearson(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def get_date_row(data, sample_date):
    normalized = pd.Timestamp(sample_date).normalize()
    exact = data.loc[data["date"].dt.normalize() == normalized]
    if not exact.empty:
        return exact
    idx = (data["date"] - pd.Timestamp(sample_date)).abs().idxmin()
    return data.loc[[idx]]


def evaluate_models(stations, data, var_name, sample_dates):
    xy = stations[["lon", "lat"]].to_numpy()
    rows = []
    station_cols = stations["StationName"].tolist()

    for date in sample_dates:
        rec = get_date_row(data, date)[station_cols]
        if rec.empty:
            continue

        values = rec.iloc[0].to_numpy(dtype=float)
        valid = np.isfinite(values)
        if valid.sum() < 5:
            continue

        xyv = xy[valid]
        zv = values[valid]
        folds = min(5, len(zv))
        if folds < 2:
            continue

        cv = KFold(n_splits=folds, shuffle=True, random_state=42)

        for model_name, fn in MODEL_FUNCS.items():
            obs_all = []
            pred_all = []
            for train_idx, test_idx in cv.split(xyv):
                try:
                    pred = fn(xyv[train_idx], zv[train_idx], xyv[test_idx])
                except Exception:
                    pred = np.full(len(test_idx), np.nan)
                obs_all.append(zv[test_idx])
                pred_all.append(pred)

            obs = np.concatenate(obs_all)
            pred = np.concatenate(pred_all)
            mask = np.isfinite(obs) & np.isfinite(pred)
            if mask.sum() == 0:
                continue

            rows.append(
                {
                    "variable": var_name,
                    "date": pd.Timestamp(date).date().isoformat(),
                    "model": model_name,
                    "rmse": math.sqrt(mean_squared_error(obs[mask], pred[mask])),
                    "correlation": pearson(obs, pred),
                    "bias": float(np.nanmean(pred[mask] - obs[mask])),
                    "n": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows, columns=["variable", "date", "model", "rmse", "correlation", "bias", "n"])


def summarize_timeseries(data, stations, var_name):
    station_cols = stations["StationName"].tolist()
    vals = data[station_cols].to_numpy(dtype=float).ravel()
    vals = vals[np.isfinite(vals)]

    annual = (
        data.assign(year=data["date"].dt.year)[station_cols + ["year"]]
        .groupby("year")
        .mean(numeric_only=True)
    )
    monthly = (
        data.assign(month=data["date"].dt.month)[station_cols + ["month"]]
        .groupby("month")
        .mean(numeric_only=True)
    )

    return {
        "variable": var_name,
        "available_station_days": int(np.isfinite(data[station_cols].to_numpy(dtype=float)).sum()),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "variance": float(np.nanvar(vals)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "annual_mean_first_year": safe_nanmean(annual.iloc[0].to_numpy(dtype=float)) if len(annual) else np.nan,
        "annual_mean_last_year": safe_nanmean(annual.iloc[-1].to_numpy(dtype=float)) if len(annual) else np.nan,
        "seasonal_peak_month": int(np.nanmean(monthly, axis=1).argmax() + 1),
        "seasonal_low_month": int(np.nanmean(monthly, axis=1).argmin() + 1),
    }


def plot_station_map(stations):
    plt.figure(figsize=(8, 5))
    plt.scatter(stations["lon"], stations["lat"], s=18, alpha=0.8)
    plt.xlim(-126, -65)
    plt.ylim(23, 51)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("USCRN stations used for CONUS interpolation")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "station_locations.png", dpi=180)
    plt.close()


def plot_model_metrics(metrics):
    if metrics.empty:
        return
    avg = metrics.groupby(["variable", "model"], as_index=False)[["rmse", "correlation"]].mean()
    avg = avg[avg["model"] != "Kriging gaussian"]
    for measure in ["rmse", "correlation"]:
        pivot = avg.pivot(index="variable", columns="model", values=measure)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title(f"Cross-validation {measure}")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / f"model_{measure}.png", dpi=180)
        plt.close()

def plot_seasonality(seasonal_long):
    if seasonal_long.empty:
        return
    for var, grp in seasonal_long.groupby("variable"):
        plt.figure(figsize=(8, 4))
        plt.plot(grp["month"], grp["value"], marker="o")
        plt.xticks(range(1, 13))
        plt.xlabel("Month")
        plt.ylabel("Spatial mean")
        plt.title(f"Mean seasonal cycle: {var}")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        safe = var.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(FIG / f"seasonality_{safe}.png", dpi=180)
        plt.close()


def plot_spatial_map(values, lons, lats, title, filename):
    try:
        Z = np.asarray(values, dtype=float).reshape(len(lats), len(lons))
        plt.figure(figsize=(8, 5))
        plt.pcolormesh(lons, lats, Z, shading="auto")
        plt.colorbar(label="Value")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=180)
        plt.close()
    except Exception:
        pass


def plot_scatter(pred_grid, era5_grid, title, filename):
    try:
        pred = np.asarray(pred_grid, dtype=float).ravel()
        true = np.asarray(era5_grid, dtype=float).ravel()
        mask = np.isfinite(pred) & np.isfinite(true)
        pred = pred[mask]
        true = true[mask]
        if len(pred) < 10:
            return

        plt.figure(figsize=(5, 5))
        plt.scatter(true, pred, alpha=0.4)

        mn = min(true.min(), pred.min())
        mx = max(true.max(), pred.max())
        plt.plot([mn, mx], [mn, mx], "r--")

        plt.xlabel("ERA5")
        plt.ylabel("Predicted")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=180)
        plt.close()
    except Exception:
        pass


def plot_rmse_timeseries(era5_df):
    if era5_df.empty:
        return

    plt.figure(figsize=(10, 4))
    for var, grp in era5_df.groupby("variable"):
        plt.plot(pd.to_datetime(grp["date"]), grp["rmse_vs_era5"], label=var)

    plt.xlabel("Date")
    plt.ylabel("RMSE")
    plt.title("RMSE vs ERA5 over time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "rmse_timeseries.png", dpi=180)
    plt.close()


def gridded_product(data, stations, grid, model_name, var_name):
    safe = var_name.lower().replace(" ", "_").replace("/", "_")
    path = GRID_DIR / f"{safe}_daily_2deg.csv"

    station_cols = stations["StationName"].tolist()
    xy = stations[["lon", "lat"]].to_numpy()
    grid_xy = grid[["lon", "lat"]].to_numpy()
    values_2d = data[station_cols].to_numpy(dtype=float)
    dates = data["date"].dt.date.astype(str).to_numpy()

    rows = []

    if model_name == "IDW":
        k = min(12, len(stations))
        tree = cKDTree(xy)
        dist, idx = tree.query(grid_xy, k=k)
        weights = 1.0 / np.maximum(dist, 1e-12) ** 2

        for i, values in enumerate(values_2d):
            neighbors = values[idx]
            valid = np.isfinite(neighbors)
            weighted = np.where(valid, neighbors * weights, 0.0)
            weight_sum = np.where(valid, weights, 0.0).sum(axis=1)
            pred = np.divide(
                weighted.sum(axis=1),
                weight_sum,
                out=np.full(len(grid), np.nan),
                where=weight_sum > 0,
            )
            out = grid.copy()
            out.insert(0, "date", dates[i])
            out["value"] = pred
            rows.append(out)
    else:
        fn = MODEL_FUNCS[model_name]
        for i, values in enumerate(values_2d):
            try:
                pred = fn(xy, values, grid_xy)
            except Exception:
                pred = idw_predict(xy, values, grid_xy)
            out = grid.copy()
            out.insert(0, "date", dates[i])
            out["value"] = pred
            rows.append(out)

    gridded = pd.concat(rows, ignore_index=True)
    gridded.to_csv(path, index=False)
    return path


def _standardize_era5_da(ds):
    var_name = list(ds.data_vars)[0]
    da = ds[var_name].squeeze(drop=True)

    if "valid_time" in da.coords:
        da = da.rename({"valid_time": "time"})
    elif "date" in da.coords:
        da = da.rename({"date": "time"})

    lat_name = "latitude" if "latitude" in da.coords else ("lat" if "lat" in da.coords else None)
    lon_name = "longitude" if "longitude" in da.coords else ("lon" if "lon" in da.coords else None)

    if lat_name is None or lon_name is None:
        raise ValueError("ERA5 file must contain latitude/longitude coordinates.")

    if lat_name == "latitude":
        da = da.rename({"latitude": "lat"})
        lat_name = "lat"
    if lon_name == "longitude":
        da = da.rename({"longitude": "lon"})
        lon_name = "lon"

    if float(da[lon_name].max()) > 180:
        da = da.assign_coords({lon_name: (((da[lon_name] + 180) % 360) - 180)}).sortby(lon_name)

    da = da.sortby(lat_name)
    da = da.sortby(lon_name)
    return da, lat_name, lon_name


def load_era5_cache():
    cache = {}
    for key, path in ERA5_FILES.items():
        if path.exists():
            cache[key] = xr.open_dataset(path)
    return cache


def build_era5_grid_for_variable(var_name, era5_cache, target_lats, target_lons):
    if var_name == "Air temperature":
        if "t2m" not in era5_cache:
            return None
        da, lat_name, lon_name = _standardize_era5_da(era5_cache["t2m"])
        da = (da - 273.15).rename({lat_name: "lat", lon_name: "lon"})
        da = da.interp(lat=target_lats, lon=target_lons, method="linear")
        return da

    if var_name == "Precipitation":
        if "tp" not in era5_cache:
            return None
        da, lat_name, lon_name = _standardize_era5_da(era5_cache["tp"])
        da = (da * 1000.0).rename({lat_name: "lat", lon_name: "lon"})
        da = da.interp(lat=target_lats, lon=target_lons, method="linear")
        return da

    if var_name == "Relative humidity":
        if "t2m" not in era5_cache or "d2m" not in era5_cache:
            return None
        t2m, lat_name, lon_name = _standardize_era5_da(era5_cache["t2m"])
        d2m, _, _ = _standardize_era5_da(era5_cache["d2m"])
        t = (t2m - 273.15).rename({lat_name: "lat", lon_name: "lon"})
        td = (d2m - 273.15).rename({lat_name: "lat", lon_name: "lon"})
        rh = 100.0 * (
            np.exp((17.625 * td) / (243.04 + td)) /
            np.exp((17.625 * t) / (243.04 + t))
        )
        rh = rh.clip(0, 100)
        rh = rh.interp(lat=target_lats, lon=target_lons, method="linear")
        return rh

    if var_name == "Soil temperature":
        if "stl1" not in era5_cache:
            return None
        da, lat_name, lon_name = _standardize_era5_da(era5_cache["stl1"])
        da = (da - 273.15).rename({lat_name: "lat", lon_name: "lon"})
        da = da.interp(lat=target_lats, lon=target_lons, method="linear")
        return da

    if var_name == "Soil moisture 10cm":
        if "swvl1" not in era5_cache:
            return None
        da, lat_name, lon_name = _standardize_era5_da(era5_cache["swvl1"])
        da = da.rename({lat_name: "lat", lon_name: "lon"})
        da = da.interp(lat=target_lats, lon=target_lons, method="linear")
        return da

    return None


def compare_with_era5(pred_grid, era5_grid):
    pred = np.asarray(pred_grid, dtype=float)
    true = np.asarray(era5_grid, dtype=float)

    n0 = min(pred.shape[0], true.shape[0])
    n1 = min(pred.shape[1], true.shape[1])
    pred = pred[:n0, :n1].ravel()
    true = true[:n0, :n1].ravel()

    mask = np.isfinite(pred) & np.isfinite(true)
    if mask.sum() < 3:
        return np.nan, np.nan

    rmse = np.sqrt(np.mean((pred[mask] - true[mask]) ** 2))
    corr = np.corrcoef(pred[mask], true[mask])[0, 1]
    return float(rmse), float(corr)


def compare_variable_with_era5(var_name, data, stations, grid, best_model, era5_cache, lats, lons):
    era5_da = build_era5_grid_for_variable(var_name, era5_cache, lats, lons)
    if era5_da is None:
        return pd.DataFrame()

    station_cols = stations["StationName"].tolist()
    xy = stations[["lon", "lat"]].to_numpy()
    grid_xy = grid[["lon", "lat"]].to_numpy()
    fn = MODEL_FUNCS.get(best_model, MODEL_FUNCS["IDW"])

    data_dates = pd.to_datetime(data["date"].dt.normalize().unique())
    era5_dates = pd.to_datetime(era5_da["time"].values).normalize()
    common_dates = sorted(set(data_dates) & set(era5_dates))

    rows = []
    example_plotted = False

    for date in common_dates:
        rec = get_date_row(data, date)[station_cols]
        if rec.empty:
            continue

        values = rec.iloc[0].to_numpy(dtype=float)
        if np.isfinite(values).sum() < 5:
            continue

        try:
            pred_flat = fn(xy, values, grid_xy)
        except Exception:
            pred_flat = idw_predict(xy, values, grid_xy)

        pred_grid = pred_flat.reshape(len(lats), len(lons))
        era5_grid = era5_da.sel(time=np.datetime64(date), method="nearest").values

        rmse, corr = compare_with_era5(pred_grid, era5_grid)
        rows.append(
            {
                "variable": var_name,
                "date": pd.Timestamp(date).date().isoformat(),
                "best_model": best_model,
                "rmse_vs_era5": rmse,
                "correlation_vs_era5": corr,
            }
        )

        if not example_plotted:
            safe_var = var_name.lower().replace(" ", "_").replace("/", "_")
            safe_date = str(pd.Timestamp(date).date())

            plot_spatial_map(
                pred_flat,
                lons,
                lats,
                f"{var_name} Interpolated ({safe_date})",
                f"{safe_var}_{safe_date}_map.png",
            )

            plot_scatter(
                pred_grid,
                era5_grid,
                f"{var_name} ERA5 vs Predicted ({safe_date})",
                f"{safe_var}_{safe_date}_scatter.png",
            )
            example_plotted = True

    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    stations = load_station_info()
    lons, lats, grid = make_grid()
    plot_station_map(stations)

    station_cols = stations["StationName"].tolist()
    stats_rows = []
    metrics = []
    seasonal_rows = []
    data_cache = {}

    sample_dates = pd.to_datetime(
        [
            "2006-01-15",
            "2008-04-15",
            "2010-07-15",
            "2012-10-15",
            "2014-01-15",
            "2016-04-15",
            "2018-07-15",
            "2020-10-15",
        ]
    )

    for var_name, path in VARIABLE_FILES.items():
        data = load_variable(path, station_cols)
        data_cache[var_name] = data

        stats_rows.append(summarize_timeseries(data, stations, var_name))
        metric = evaluate_models(stations, data, var_name, sample_dates)
        if not metric.empty:
            metrics.append(metric)

        monthly = (
            data.assign(month=data["date"].dt.month)[station_cols + ["month"]]
            .groupby("month")
            .mean(numeric_only=True)
        )
        for month, row in monthly.iterrows():
            seasonal_rows.append(
                {
                    "variable": var_name,
                    "month": int(month),
                    "value": float(np.nanmean(row.to_numpy(dtype=float))),
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    seasonal_df = pd.DataFrame(seasonal_rows)

    if len(metrics) == 0:
        raise ValueError(
            "No evaluation metrics were computed. "
            "Check that station names match the CSV columns, the dates exist in the files, "
            "and the files contain valid numeric data."
        )

    metrics_df = pd.concat(metrics, ignore_index=True)

    best = metrics_df.groupby(["variable", "model"], as_index=False)[["rmse", "correlation"]].mean()
    best = best.sort_values(["variable", "rmse", "correlation"], ascending=[True, True, False])
    best_models = best.groupby("variable", as_index=False).first()
    best_model_lookup = best_models.set_index("variable")["model"].to_dict()

    plot_model_metrics(metrics_df)
    plot_seasonality(seasonal_df)

    grid_paths = {}
    for var_name, data in data_cache.items():
        model = best_model_lookup.get(var_name, "IDW")
        grid_paths[var_name] = gridded_product(data, stations, grid, model, var_name)

    era5_cache = load_era5_cache()
    era5_results = []

    for var_name, data in data_cache.items():
        if var_name not in best_model_lookup:
            continue
        era5_df = compare_variable_with_era5(
            var_name=var_name,
            data=data,
            stations=stations,
            grid=grid,
            best_model=best_model_lookup[var_name],
            era5_cache=era5_cache,
            lats=lats,
            lons=lons,
        )
        if not era5_df.empty:
            era5_results.append(era5_df)

    if era5_results:
        era5_df = pd.concat(era5_results, ignore_index=True)
        era5_df.to_csv(OUT / "era5_comparison.csv", index=False)
        plot_rmse_timeseries(era5_df)
    else:
        era5_df = pd.DataFrame()

    stats_df.to_csv(OUT / "statistical_summary.csv", index=False)
    metrics_df.to_csv(OUT / "cross_validation_metrics.csv", index=False)
    best_models.to_csv(OUT / "best_models.csv", index=False)
    seasonal_df.to_csv(OUT / "seasonality_monthly_means.csv", index=False)

    ks_rows = []
    for var_name, data in data_cache.items():
        model = best_model_lookup.get(var_name, "IDW")
        station_cols = stations["StationName"].tolist()
        xy = stations[["lon", "lat"]].to_numpy()
        grid_xy = grid[["lon", "lat"]].to_numpy()
        fn = MODEL_FUNCS.get(model, MODEL_FUNCS["IDW"])

        selected = data[data["date"].dt.day.eq(15)].iloc[::12].head(12)
        selected_values = selected[station_cols].to_numpy(dtype=float)

        station_samples = []
        grid_samples = []
        for values in selected_values:
            try:
                pred = fn(xy, values, grid_xy)
            except Exception:
                pred = idw_predict(xy, values, grid_xy)
            station_samples.extend(values[np.isfinite(values)])
            grid_samples.extend(pred[np.isfinite(pred)])

        if len(station_samples) and len(grid_samples):
            stat, pval = ks_2samp(station_samples, grid_samples)
            ks_rows.append(
                {
                    "variable": var_name,
                    "best_model": model,
                    "ks_statistic": stat,
                    "ks_pvalue": pval,
                }
            )

    ks_df = pd.DataFrame(ks_rows, columns=["variable", "best_model", "ks_statistic", "ks_pvalue"])
    ks_df.to_csv(OUT / "ks_distribution_tests.csv", index=False)

    metadata = {
        "domain": "CONUS",
        "grid_resolution_degrees": 2.0,
        "n_stations_used": int(len(stations)),
        "n_grid_cells": int(len(grid)),
        "evaluation": "5-fold station cross-validation; ERA5 comparison included when matching files are present",
        "variables": list(VARIABLE_FILES),
        "era5_year": ERA5_YEAR,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()