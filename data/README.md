# Data Folder

This folder contains all input datasets used for the geospatial interpolation and analysis of climate variables over the contiguous United States (CONUS).

---

## Contents

### 1. Station Metadata
- **All_CONUS_Station_Information_V2.csv**  
  Contains metadata for USCRN stations, including:
  - Station name  
  - Latitude and longitude  
  - Location information  

---

### 2. Climate Variables (USCRN)

These files contain daily observations from USCRN stations (2006–2021):

- **USCRN_AirTemperature_2006_2021.csv**
- **USCRN_Precipitation_2006_2021.csv**
- **USCRN_RelativeHumidity_2006_2021.csv**
- **USCRN_SoilMoisture10cm_2006_2021.csv**
- **USCRN_SoilTemperature_2006_2021.csv**

Each file:
- Contains a `date` column  
- Contains station-wise columns (station names as headers)  
- Values represent daily observations  

---

### 3. ERA5 Reanalysis Data

ERA5 data is used for validation (year: 2015):

- **era5_t2m_2015.nc** — 2m temperature  
- **era5_tp_2015.nc** — total precipitation  
- **era5_d2m_2015.nc** — dewpoint temperature  
- **era5_stl1_2015.nc** — soil temperature (level 1)  
- **era5_swvl1_2015.nc** — soil moisture (layer 1)  

---
