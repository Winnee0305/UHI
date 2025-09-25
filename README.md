## Urban Heat Island (UHI) Prediction Pipeline

This repository contains a geospatial machine learning pipeline to predict Urban Heat Island (UHI) indices from multi-sensor satellite imagery (Sentinel‑2 and Landsat 8) using Microsoft Planetary Computer data and a Random Forest model.

### Features

- Fetches and composites Sentinel‑2 and Landsat scenes over an AOI derived from training coordinates
- Builds a cloud‑reduced median mosaic, computes key indices (NDVI, NDWI, NDBI, MSI)
- Stacks selected Sentinel‑2 and Landsat bands into a single GeoTIFF (`combined.tiff`)
- Samples raster values at training and test coordinates
- Trains a `RandomForestRegressor` and generates `submission.csv`

### Repository Structure

- `main.py`: End‑to‑end pipeline (data retrieval, preprocessing, feature engineering, training, inference, submission)
- `config.py`: Global configuration (band lists, temporal window, bounding box derivation)
- `Training_data_uhi_index.csv`: Labeled training data with `Latitude`, `Longitude`, and `UHI Index`
- `Submission_template.csv`: Template file with `Latitude` and `Longitude` for inference
- `combined.tiff`: Multi-band GeoTIFF produced by the pipeline (created on first run)
- `submission.csv`: Output predictions (created by `main.py`)
- `*.ipynb`: Exploration notebooks for Sentinel‑2 and Landsat workflows

### Requirements

Tested on Python 3.9+ (Anaconda recommended). Install the following packages:

```bash
pip install matplotlib seaborn numpy pandas xarray rioxarray geopandas rasterio pillow pyproj scikit-learn pystac-client planetary-computer tqdm stackstac odc-stac
```

If using conda, ensure GDAL/GEOS/PROJ stack is consistent (e.g., `conda-forge` channel):

```bash
conda create -n uhi python=3.9 -y
conda activate uhi
conda install -c conda-forge gdal rasterio geopandas rioxarray pyproj xarray -y
pip install matplotlib seaborn scikit-learn tqdm pystac-client planetary-computer stackstac odc-stac
```

### Data Sources

- Microsoft Planetary Computer STAC API for Sentinel‑2 L2A and Landsat Collection 2 Level‑2
- Training CSV (`Training_data_uhi_index.csv`) defines the Area of Interest and ground truth

### How It Works (High Level)

1. Read training and submission CSVs; compute AOI bounds from training coordinates.
2. Query Planetary Computer STAC for Sentinel‑2 (cloud < 30%) and Landsat 8 (cloud < 50%) within `2021-06-01/2021-09-01`.
3. Load scenes via `odc.stac.stac_load`, reproject to EPSG:4326, and compute a cloud‑reduced median composite for Sentinel‑2.
4. Scale Landsat reflectance and LST bands to physical values.
5. Write a 19‑band `combined.tiff` stacking Sentinel‑2 and Landsat bands.
6. Sample band values at training/test coordinates; compute NDVI/NDWI/NDBI/MSI features.
7. Train `RandomForestRegressor`, evaluate, transform test features, and write `submission.csv`.

### Usage

From the project root:

```bash
# Optional: run with your Anaconda python
/opt/anaconda3/bin/python \
  "/Users/winnee/Study/EY Challenge/UHI/main.py"

# Or simply
python main.py
```

Outputs:

- `combined.tiff`: 19‑band GeoTIFF of the AOI
- `submission.csv`: Predictions for the submission template coordinates
- `img/*.png`: Diagnostic plots (created during the run)

### Important Notes

- Filenames are referenced in code as lowercase (`training_data_uhi_index.csv`, `submission_template.csv`). If your OS is case‑sensitive, ensure the CSV filenames match or adjust the names in `main.py`/`config.py`.
- The script assumes write access to create `combined.tiff`, `submission.csv`, and an `img/` folder. If `img/` doesn’t exist, create it beforehand: `mkdir -p img`.
- Access to Planetary Computer public datasets does not require an API key, but be mindful of usage policies.
