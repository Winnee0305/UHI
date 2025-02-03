# Install module: 
# pip install matplotlib seaborn numpy pandas xarray rioxarray geopandas rasterio pillow pyproj scikit-learn pystac-client planetary-computer tqdm

# Run the code: (using anaconda)
# /opt/anaconda3/bin/python "/Users/winnee/Study/EY Challenge/UHI/main.py"

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Science
import numpy as np
import pandas as pd

# Multi-dimensional arrays and datasets
import xarray as xr

# Geospatial raster data handling
import rioxarray as rxr

# Geospatial data analysis
import geopandas as gpd

# Geospatial operations
import rasterio
from rasterio import windows  
from rasterio import features  
from rasterio import warp
from rasterio.warp import transform_bounds 
from rasterio.windows import from_bounds 

# Image Processing
from PIL import Image

# Coordinate transformations
from pyproj import Proj, Transformer, CRS

# Feature Engineering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Planetary Computer Tools
import pystac_client
import planetary_computer as pc
from pystac.extensions.eo import EOExtension as eo

# Others
import os
from tqdm import tqdm

# Import common GIS tools
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import rioxarray as rio
import rasterio
from matplotlib.cm import RdYlGn,jet,RdBu

# Import Planetary Computer tools
import stackstac
import pystac_client
import planetary_computer 
from odc.stac import stac_load

import json

# Global vairable for file names
training_data_filename = "Training_data_uhi_index.csv"
submission_filename = "Submission_template.csv"

ground_df = pd.read_csv(training_data_filename)
ground_df.head()
print(ground_df.shape)

submission = pd.read_csv(submission_filename)
submission.head()
print(submission.shape)

# Extract the minimum and maximum latitude and longitude from the dataset
min_lat = ground_df['Latitude'].min()
max_lat = ground_df['Latitude'].max()
min_lon = ground_df['Longitude'].min()
max_lon = ground_df['Longitude'].max()

lower_left = (min_lat, min_lon)
upper_right = (max_lat, max_lon)

# Define the bounding box using the min/max values
# This will represent the area where your data was collected (Manhattan and the Bronx)
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

# Print the defined bounds
print(f"Area of Interest Bounding Box: {bounds}")

# Define the time window
time_window = "2021-06-01/2021-09-01"

# Winnee implementation
# def createBoundsForSinglePoint (latitude, longitude, margin):
#     point = (latitude, longitude)
#     lower_left = (point[0] - margin, point[1] - margin)  
#     upper_right = (point[0] + margin, point[1] + margin) 
#     bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
#     return bounds

# # Define the time window
# margin = 0.001
# time_window = "2021-06-01/2021-09-01"

stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

search = stac.search(
    bbox=bounds,
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)

items = list(search.get_items())
print('This is the number of scenes that touch our region:',len(items))

signed_items = [planetary_computer.sign(item).to_dict() for item in items]

# signed_items = [planetary_computer.sign(item) for item in items]


resolution = 10  # meters per pixel 
scale = resolution / 111320.0 # degrees per pixel for crs=4326 

data = stac_load(
    items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds,
)

print(data)

# Plot sample images from the time series
plot_data = data[["B04","B03","B02"]].to_array()
plot_data.plot.imshow(col='time', col_wrap=4, robust=True, vmin=0, vmax=2500)
plt.show()

# Plot an RGB image for a single date
fig, ax = plt.subplots(figsize=(6,6), dpi=150)
plot_data.isel(time=7).plot.imshow(robust=True, ax=ax, vmin=0, vmax=2500)
ax.set_title("RGB Single Date: July 24, 2021")
ax.axis('off')
plt.show()

median = data.median(dim="time").compute()

# Plot an RGB image for the median composite or mosaic
# Notice how this new image is void of clouds due to statistical filtering
fig, ax = plt.subplots(figsize=(6,6), dpi=150)
median[["B04", "B03", "B02"]].to_array().plot.imshow(robust=True, ax=ax, vmin=0, vmax=2500)
ax.set_title("RGB Median Composite")
ax.axis('off')
plt.show()

# Calculate NDVI for the median mosaic
ndvi_median = (median.B08-median.B04)/(median.B08+median.B04)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
ndvi_median.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn")
plt.title("Median NDVI")
plt.axis('off')
plt.show()

# Calculate NDBI for the median mosaic
ndbi_median = (median.B11-median.B08)/(median.B11+median.B08)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
ndbi_median.plot.imshow(vmin=-0.1, vmax=0.1, cmap="jet")
plt.title("Median NDBI")
plt.axis('off')
plt.show()

# Calculate NDWI for the median mosaic
ndwi_median = (median.B03-median.B08)/(median.B03+median.B08)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
ndwi_median.plot.imshow(vmin=-0.3, vmax=0.3, cmap="RdBu")
plt.title("Median NDWI")
plt.axis('off')
plt.show()

filename = "S2_sample.tiff"

# We will pick a single time slice from the time series (time=7) 
# This time slice is the date of July 24, 2021
data_slice = data.isel(time=7)

# Calculate the dimensions of the file
# height = median.dims["latitude"]
# width = median.dims["longitude"]
height = data_slice.dims["latitude"]
width = data_slice.dims["longitude"]

# Define the Coordinate Reference System (CRS) to be common Lat-Lon coordinates
# Define the tranformation using our bounding box so the Lat-Lon information is written to the GeoTIFF
gt = rasterio.transform.from_bounds(lower_left[1],lower_left[0],upper_right[1],upper_right[0],width,height)
data_slice.rio.write_crs("epsg:4326", inplace=True)
data_slice.rio.write_transform(transform=gt, inplace=True);

# Create the GeoTIFF output file using the defined parameters 
with rasterio.open(filename,'w',driver='GTiff',width=width,height=height,
                   crs='epsg:4326',transform=gt,count=4,compress='lzw',dtype='float64') as dst:
    dst.write(data_slice.B01,1)
    dst.write(data_slice.B04,2)
    dst.write(data_slice.B06,3) 
    dst.write(data_slice.B08,4)
    dst.close()

# Show the location and size of the new output file
for file in os.listdir('.'):
    if file.endswith('.tiff'):
        print(file)