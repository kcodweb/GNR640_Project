# Geospatial Interpolation of Climate Variables (CONUS)

This project generates gridded climate datasets over the contiguous United States (CONUS) using station-based observations from the USCRN network and evaluates interpolation methods against ERA5 reanalysis data.

---

## Objective

- Convert station-based climate observations into a regular 2° × 2° grid  
- Compare multiple interpolation techniques  
- Validate results using ERA5 reanalysis data  
- Analyze spatial and temporal patterns  

---

## Repository Structure

```
project/
│
├── data/                 # Input datasets (USCRN + ERA5)
├── project_outputs/      # All outputs (CSV, figures, results)
├── main.py               # Main pipeline script
```

---

## Data Sources

### 1. USCRN Station Data (2006–2021)
- Air temperature  
- Precipitation  
- Relative humidity  
- Soil moisture (10 cm)  
- Soil temperature  

### 2. ERA5 Reanalysis Data (2015)
- Used for external validation  
- NetCDF format  

---

## Methodology

### Interpolation Methods
- Inverse Distance Weighting (IDW)  
- Linear interpolation  
- Radial Basis Function (RBF)  
- Kriging (spherical, circular, gaussian)

### Evaluation Metrics
- RMSE (Root Mean Square Error)  
- Correlation coefficient  
- Bias  

### Validation
- 5-fold cross-validation  
- Comparison with ERA5  

---

## Outputs

All results are saved in:

```
project_outputs/
```

### Includes:
- Gridded datasets (`gridded_daily_2deg/`)
- Cross-validation metrics (`cross_validation_metrics.csv`)
- Best models (`best_models.csv`)
- ERA5 comparison (`era5_comparison.csv`)
- KS tests (`ks_distribution_tests.csv`)
- Statistical summaries (`statistical_summary.csv`)
- Seasonal analysis (`seasonality_monthly_means.csv`)
- Figures (`figures/`)

---

## Figures

Generated plots include:
- Model performance (RMSE, correlation)
- Spatial interpolation maps
- ERA5 vs predicted scatter plots
- RMSE time series
- Seasonal cycles
- Station locations

---

## How to Run

1. Install dependencies:
```
pip install numpy pandas matplotlib xarray scipy scikit-learn pykrige netCDF4
```

2. Place all input files in the `data/` folder

3. Run the script:
```
python main.py
```

---

## Report

The report summarizes:
- Methodology  
- Model comparison  
- Spatial and temporal analysis  
- ERA5 validation results  

All results, including figures and tables, are available in the `project_outputs` folder.

---

## 👨‍💻 Author

Karan Bansal (24b3003)
