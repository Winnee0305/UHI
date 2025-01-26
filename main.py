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

# Global vairable for file names
training_data_filename = "Training_data_uhi_index.csv"
submission_filename = "Submission_template.csv"

ground_df = pd.read_csv(training_data_filename)
ground_df.head()
print(ground_df.shape)

submission = pd.read_csv(submission_filename)
submission.head()
print(submission.shape)


def createBoundsForSinglePoint (latitude, longitude, margin):
    point = (latitude, longitude)
    lower_left = (point[0] - margin, point[1] - margin)  
    upper_right = (point[0] + margin, point[1] + margin) 
    bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])
    return bounds

# Define the time window
margin = 0.001
time_window = "2021-06-01/2021-09-01"

stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")


search = stac.search(
    bbox=createBoundsForSinglePoint(40.81310667,-73.90916667,margin),
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)

items = list(search.get_items())
print('This is the number of scenes that touch our region:',len(items))

signed_items = [planetary_computer.sign(item).to_dict() for item in items]

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
    bbox=createBoundsForSinglePoint(40.81310667,-73.90916667,margin),
)