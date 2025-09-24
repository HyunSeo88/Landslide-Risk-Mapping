# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **Landslide Risk Analysis** project focusing on South Korea's southeastern regions (Gyeongsangbuk-do, Gyeongsangnam-do, Daegu, Ulsan, Busan). The project integrates multiple geospatial datasets and meteorological data for comprehensive landslide susceptibility analysis.

## Project Structure

```
D:\Landslide/
├── data/
│   ├── LDAPS/                    # Weather prediction data processing
│   │   ├── ldaps_load_grib.py   # GRIB data download & NetCDF conversion
│   │   ├── ldaps_process.py     # NetCDF to GeoTIFF processing
│   │   ├── GRIB/                # Raw GRIB files
│   │   ├── NetCDF/              # Processed NetCDF files
│   │   └── ldaps_tif/           # Final GeoTIFF outputs
│   ├── utils/                   # Utility scripts and notebooks
│   │   ├── 좌표계변환.ipynb      # CRS conversion utilities
│   │   └── 연구지역클리핑.ipynb   # Study area mask generation
│   ├── 연구지역클리핑/           # Study area masks
│   │   ├── study_area_mask.shp  # Unified study area boundary
│   │   └── selected_regions.shp # Individual administrative regions
│   ├── 행정구역경계_wgs84/       # Administrative boundaries (WGS84)
│   ├── DEM/                     # Digital Elevation Model data
│   ├── 수치지질도_25만축척_전국/  # 1:250,000 scale geological map
│   ├── merge_landcover/         # Land cover data
│   └── 자료조사/                # Research and data collection
├── ldaps.yml                    # Conda environment configuration
└── CLAUDE.md                    # This file
```

## Development Environment

**Primary Environment:** `snap` conda environment
- **Python:** 3.9.23
- **Key Geospatial Libraries:**
  - GDAL 3.11.3 (geospatial data processing)
  - GeoPandas, Rasterio (vector/raster processing)  
  - xarray, NetCDF4 (meteorological data)
  - PyTorch 2.7.1+cu118 (ML capabilities)
- **Meteorological Data:**
  - eccodes, cfgrib (GRIB file processing)
  - python-eccodes (ECMWF data handling)

## Data Processing Workflows

### 1. LDAPS Weather Data Processing
- **Input:** KMA GRIB files via API
- **Processing Chain:** GRIB → NetCDF → GeoTIFF
- **Coordinate System:** Lambert Conformal Conic (LCC)
  ```
  +proj=lcc +lat_1=30 +lat_2=60 +lat_0=38 +lon_0=126 +datum=WGS84 +units=m +no_defs
  ```
- **Output Products:**
  - 5-day accumulated precipitation
  - Daily peak rainfall intensity

### 2. Coordinate System Management
- **Standard CRS:** EPSG:4326 (WGS84) for integration
- **Native LDAPS CRS:** LCC projection for Korean Peninsula
- **Conversion Utilities:** Automated batch processing for .shp/.tif files

### 3. Study Area Definition
- **Target Regions:** 5 administrative areas in SE Korea
- **Mask Generation:** Automated boundary extraction and unification
- **Products:** Individual and unified area masks

## Key Scripts and Functions

### LDAPS Data Processing
- `ldaps_load_grib.py`: Downloads GRIB files, converts to NetCDF with regional masking
- `ldaps_process.py`: Processes NetCDF to analysis-ready GeoTIFF format
- Uses Korean Meteorological Administration API with service key authentication

### Geospatial Utilities
- `좌표계변환.ipynb`: Directory-recursive CRS conversion
- `연구지역클리핑.ipynb`: Administrative boundary processing and mask generation

## Data Sources
- **Meteorological:** KMA LDAPS (Local Data Assimilation and Prediction System)
- **Administrative Boundaries:** Korean administrative divisions
- **Topographic:** DEM data for elevation analysis
- **Geological:** 1:250,000 scale national geological survey
- **Land Cover:** National land cover classification

## Current Status
- ✅ LDAPS weather data processing pipeline
- ✅ Coordinate system standardization utilities  
- ✅ Study area boundary definition
- ✅ Basic geospatial data collection
- 🔄 Integration of all datasets pending
- 🔄 Landslide susceptibility modeling pending

## Development Notes
- All processing maintains geospatial metadata integrity
- Automated error handling for data gaps and API failures
- Memory-efficient processing for large meteorological datasets
- Standardized file naming conventions for temporal data series