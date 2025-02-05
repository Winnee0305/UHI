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

training_data_filename = "training_data_uhi_index.csv"

# File to store dataset

s2_tiff = "s2.tiff"

lst_tiff = "lst.tiff"

combined_tiff = "combined.tiff"

submission_template = "submission_template.csv"

training_df = pd.read_csv(training_data_filename, header=0, sep=',')
training_df.head()
print("Size of the training data: " , training_df.shape)

submission = pd.read_csv(submission_template)
submission.head()
print("Size of the submission data: " , submission.shape)

# Extract the minimum and maximum latitude and longitude from the dataset
min_lat = training_df['Latitude'].min()
max_lat = training_df['Latitude'].max()
min_lon = training_df['Longitude'].min()
max_lon = training_df['Longitude'].max()

lower_left = (min_lat, min_lon)
upper_right = (max_lat, max_lon)

# Define the bounding box using the min/max values
bounds = (lower_left[1], lower_left[0], upper_right[1], upper_right[0])

# Print the defined bounds
print(f"Area of Interest Bounding Box: {bounds}")

# Define the time window
time_window = "2021-06-01/2021-09-01"

stac = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

sentinel_search = stac.search(
    bbox=bounds,
    datetime=time_window,
    collections=["sentinel-2-l2a"],
    query={"eo:cloud_cover": {"lt": 30}},
)


sentinel_items = list(sentinel_search.get_items())
print('This is the number of scenes that touch our region:',len(sentinel_items))


sentinel_signed_items = [planetary_computer.sign(item).to_dict() for item in sentinel_items]

sentinel_resolution = 10  # meters per pixel 
sentinel_scale = sentinel_resolution / 111320.0 # degrees per pixel for crs=4326 

landsat_search = stac.search(
    bbox=bounds, 
    datetime=time_window,
    collections=["landsat-c2-l2"],
    query={"eo:cloud_cover": {"lt": 50},"platform": {"in": ["landsat-8"]}},
)

landsat_items = list(landsat_search.get_items())
print('This is the number of scenes that touch our region:',len(landsat_items))

landsat_signed_items = [planetary_computer.sign(item).to_dict() for item in landsat_items]

landsat_resolution = 30  # meters per pixel 
landsat_scale = landsat_resolution / 111320.0 # degrees per pixel for crs=4326 

sentinel_data = stac_load(
    sentinel_items,
    bands=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=sentinel_scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds,
)

landsat_data1 = stac_load(
    landsat_items,
    bands=["blue","green","red","nir08","swir16","swir22","coastal"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=landsat_scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)

landsat_data2 = stac_load(
    landsat_items,
    bands=["lwir11"],
    crs="EPSG:4326", # Latitude-Longitude
    resolution=landsat_scale, # Degrees
    chunks={"x": 2048, "y": 2048},
    dtype="uint16",
    patch_url=planetary_computer.sign,
    bbox=bounds
)

# Plot sample images from the time series
plot_sentinel_data = sentinel_data[["B04","B03","B02"]].to_array()
plot_sentinel_data.plot.imshow(col='time', col_wrap=4, robust=True, vmin=0, vmax=2500)
plt.savefig("img/sentinel_images.png", dpi=300, bbox_inches='tight')

# Plot an RGB image for a single date
fig, ax = plt.subplots(figsize=(6,6), dpi=150)
plot_sentinel_data.isel(time=7).plot.imshow(robust=True, ax=ax, vmin=0, vmax=2500)
ax.set_title("RGB Single Date: July 24, 2021")
ax.axis('off')
plt.savefig("img/sentinel_single_date.png", dpi=300, bbox_inches='tight')

sentinel_median = sentinel_data.median(dim="time").compute()

# Plot an RGB image for the median composite or mosaic
# Notice how this new image is void of clouds due to statistical filtering
fig, ax = plt.subplots(figsize=(6,6), dpi=150)
sentinel_median[["B04", "B03", "B02"]].to_array().plot.imshow(robust=True, ax=ax, vmin=0, vmax=2500)
ax.set_title("RGB Median Composite")
ax.axis('off')
plt.savefig("img/sentinel_median.png", dpi=300, bbox_inches='tight')

# Calculate NDVI for the median mosaic
sentinel_ndvi_median = (sentinel_median.B08-sentinel_median.B04)/(sentinel_median.B08+sentinel_median.B04)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
sentinel_ndvi_median.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn")
plt.title("Median NDVI")
plt.axis('off')
plt.savefig("img/sentinel_ndvi.png", dpi=300, bbox_inches='tight')

# Calculate NDBI for the median mosaic
sentinel_ndbi_median = (sentinel_median.B11-sentinel_median.B08)/(sentinel_median.B11+sentinel_median.B08)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
sentinel_ndbi_median.plot.imshow(vmin=-0.1, vmax=0.1, cmap="jet")
plt.title("Median NDBI")
plt.axis('off')
plt.savefig("img/sentinel_ndbi.png", dpi=300, bbox_inches='tight')

# Calculate NDWI for the median mosaic
sentinel_ndwi_median = (sentinel_median.B03-sentinel_median.B08)/(sentinel_median.B03+sentinel_median.B08)

fig, ax = plt.subplots(figsize=(7,6), dpi=150)
sentinel_ndwi_median.plot.imshow(vmin=-0.3, vmax=0.3, cmap="RdBu")
plt.title("Median NDWI")
plt.axis('off')
plt.savefig("img/sentinel_ndwi.png", dpi=300, bbox_inches='tight')

# Persist the data in memory for faster operations
landsat_data1 = landsat_data1.persist()
landsat_data2 = landsat_data2.persist()

# Scale Factors for the RGB and NIR bands 
scale1 = 0.0000275 
offset1 = -0.2 
landsat_data1 = landsat_data1.astype(float) * scale1 + offset1

# Scale Factors for the Surface Temperature band
scale2 = 0.00341802 
offset2 = 149.0 
kelvin_celsius = 273.15 # convert from Kelvin to Celsius
landsat_data2 = landsat_data2.astype(float) * scale2 + offset2 - kelvin_celsius

plot_landsat_data1 = landsat_data1[["red","green","blue"]].to_array()
plot_landsat_data1.plot.imshow(col='time', col_wrap=4, robust=True, vmin=0, vmax=0.25)
plt.savefig("img/landsat_images.png", dpi=300, bbox_inches='tight')


# Pick one of the scenes above (numbering starts with 0)
landsat_scene = 2

# Plot an RGB Real Color Image for a single date
fig, ax = plt.subplots(figsize=(9,10))
landsat_data1.isel(time=landsat_scene)[["red", "green", "blue"]].to_array().plot.imshow(robust=True, ax=ax, vmin=0.0, vmax=0.25)
ax.set_title("RGB Real Color")
ax.axis('off')
plt.savefig("img/landsat_single_date.png", dpi=300, bbox_inches='tight')


# Calculate NDVI for the median mosaic
landsat_ndvi_data = (landsat_data1.isel(time=landsat_scene).nir08-landsat_data1.isel(time=landsat_scene).red)/(landsat_data1.isel(time=landsat_scene).nir08+landsat_data1.isel(time=landsat_scene).red)

fig, ax = plt.subplots(figsize=(11,10))
landsat_ndvi_data.plot.imshow(vmin=0.0, vmax=1.0, cmap="RdYlGn")
plt.title("Vegetation Index = NDVI")
plt.axis('off')
plt.savefig("img/landsat_ndvi.png", dpi=300, bbox_inches='tight')

landsat_lwir11_data = landsat_data2.isel(time=landsat_scene).lwir11

fig, ax = plt.subplots(figsize=(11,10))
landsat_lwir11_data.plot.imshow(vmin=20.0, vmax=45.0, cmap="jet")
plt.title("Land Surface Temperature (LST)")
plt.axis('off')
plt.savefig("img/landsat_lst.png", dpi=300, bbox_inches='tight')

# We will pick a single time slice from the time series (time=7) 
# This time slice is the date of July 24, 2021
# sentinel_data_slice = sentinel_median.isel(time=7)
sentinel_data_slice = sentinel_median

height = sentinel_data_slice.dims["latitude"]
width = sentinel_data_slice.dims["longitude"]

# Only select one of the time slices to output
landsat_data1_slice = landsat_data1.isel(time=landsat_scene)
landsat_data2_slice = landsat_data2.isel(time=landsat_scene)

landsat_data1_height =  landsat_data1_slice.dims["latitude"]
landsat_data1_width =  landsat_data1_slice.dims["longitude"]

landsat_data2_height =  landsat_data2_slice.dims["latitude"]
landsat_data2_width =  landsat_data2_slice.dims["longitude"]

# Define the Coordinate Reference System (CRS) to be common Lat-Lon coordinates
# Define the tranformation using our bounding box so the Lat-Lon information is written to the GeoTIFF
gt = rasterio.transform.from_bounds(lower_left[1],lower_left[0],upper_right[1],upper_right[0],width,height)

sentinel_data_slice.rio.write_crs("epsg:4326", inplace=True)
sentinel_data_slice.rio.write_transform(transform=gt, inplace=True); 

landsat_data1_slice.rio.write_crs("epsg:4326", inplace=True)
landsat_data1_slice.rio.write_transform(transform=gt, inplace=True);

landsat_data2_slice.rio.write_crs("epsg:4326", inplace=True)
landsat_data2_slice.rio.write_transform(transform=gt, inplace=True);

# Create the GeoTIFF output file using the defined parameters 
with rasterio.open("combined.tiff",'w',driver='GTiff',width=width,height=height,
                   crs='epsg:4326',transform=gt,count=19,compress='lzw',dtype='float64') as dst:
    dst.write(sentinel_data_slice.B01,1)
    dst.write(sentinel_data_slice.B02,2)
    dst.write(sentinel_data_slice.B03,3)
    dst.write(sentinel_data_slice.B04,4)
    dst.write(sentinel_data_slice.B05,5)
    dst.write(sentinel_data_slice.B06,6)
    dst.write(sentinel_data_slice.B07,7)
    dst.write(sentinel_data_slice.B08,8)
    dst.write(sentinel_data_slice.B8A,9)
    dst.write(sentinel_data_slice.B11,10)
    dst.write(sentinel_data_slice.B12,11)
    dst.write(landsat_data1_slice.nir08,12)
    dst.write(landsat_data1_slice.red,13)
    dst.write(landsat_data1_slice.green,14)
    dst.write(landsat_data1_slice.blue,15)
    dst.write(landsat_data1_slice.swir16,16)
    dst.write(landsat_data1_slice.swir22,17)
    dst.write(landsat_data1_slice.coastal,18)
    dst.write(landsat_data2_slice.lwir11,19)
    dst.close()

# Read the bands from the GeoTIFF file
with rasterio.open(combined_tiff) as src1:
    band1 = src1.read(1)  # Band [B01]
    band2 = src1.read(2)  # Band [B02]
    band3 = src1.read(3)  # Band [B03]
    band4 = src1.read(4)  # Band [B04]
    band5 = src1.read(5)  # Band [B05]
    band6 = src1.read(6)  # Band [B06]
    band7 = src1.read(7)  # Band [B07]
    band8 = src1.read(8)  # Band [B08]
    band9 = src1.read(9)  # Band [B08A]
    band10 = src1.read(10)  # Band [B11]
    band11 = src1.read(11)  # Band [B12]
    band12 = src1.read(12)  # Band [NIR08]
    band13 = src1.read(13)  # Band [Red]
    band14 = src1.read(14)  # Band [Green]
    band15 = src1.read(15)  # Band [Blue]
    band16 = src1.read(16)  # Band [SWIR16]
    band17 = src1.read(17)  # Band [SWIR22]
    band18 = src1.read(18)  # Band [Coastal]
    band19 = src1.read(19)  # Band [LWIR11]

# Plot the bands in a 2x3 grid
fig, axes = plt.subplots(4, 3, figsize=(10, 10))

# Flatten the axes for easier indexing
axes = axes.flatten()

# Plot the first band (B01)
im1 = axes[0].imshow(band1, cmap='viridis')
axes[0].set_title('Band [B01]')
fig.colorbar(im1, ax=axes[0])

# Plot the second band (B04)
im2 = axes[1].imshow(band2, cmap='viridis')
axes[1].set_title('Band [B04]')
fig.colorbar(im2, ax=axes[1])

# Plot the third band (B06)
im3 = axes[2].imshow(band3, cmap='viridis')                 
axes[2].set_title('Band [B06]')
fig.colorbar(im3, ax=axes[2])

# Plot the fourth band (B08)
im4 = axes[3].imshow(band4, cmap='viridis')
axes[3].set_title('Band [B08]')
fig.colorbar(im4, ax=axes[3])

# Plot the fifth band (NIR08)
im5 = axes[4].imshow(band5, cmap='viridis')
axes[4].set_title('Band [NIR08]')
fig.colorbar(im5, ax=axes[4])

# Plot the sixth band (Red)
im6 = axes[5].imshow(band6, cmap='viridis')
axes[5].set_title('Band [Red]')
fig.colorbar(im6, ax=axes[5])

# Plot the fifth band (LWIR11)
im7 = axes[6].imshow(band7, cmap='viridis')
axes[6].set_title('Band [LWIR11]')
fig.colorbar(im7, ax=axes[6])

# Plot the sixth band (B03)
im8 = axes[7].imshow(band8, cmap='viridis')
axes[7].set_title('Band [B03]')
fig.colorbar(im8, ax=axes[7])

# Plot the fifth band (B11)
im9 = axes[8].imshow(band9, cmap='viridis')
axes[8].set_title('Band [B11]')
fig.colorbar(im9, ax=axes[8])

plt.tight_layout()
plt.savefig("img/bands.png", dpi=300, bbox_inches='tight')


# Extracts satellite band values from a GeoTIFF based on coordinates from a csv file and returns them in a DataFrame.

def map_satellite_data(tiff_path, csv_path):
    
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    tiff_crs = data.rio.crs

    # Read the Excel file using pandas
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values

    # 3. Convert lat/long to the GeoTIFF's CRS
    # Create a Proj object for EPSG:4326 (WGS84 - lat/long) and the GeoTIFF's CRS
    proj_wgs84 = Proj(init='epsg:4326')  # EPSG:4326 is the common lat/long CRS
    proj_tiff = Proj(tiff_crs)
    
    # Create a transformer object
    transformer = Transformer.from_proj(proj_wgs84, proj_tiff)

    B01_values = []
    B02_values = []
    B03_values = []
    B04_values = []
    B05_values = []
    B06_values = []
    B07_values = []
    B08_values = []
    B08A_values = []
    B11_values = []
    B12_values = []
    NIR08_values = []
    Red_values = []
    Green_values = []
    Blue_values = []
    SWIR16_values = []
    SWIR22_values = []
    Coastal_values = []
    LWIR11_values = []

# Iterate over the latitudes and longitudes, and extract the corresponding band values
    for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
    # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names from data.coords)
    
        B01_value = data.sel(x=lon, y=lat,  band=1, method="nearest").values
        B01_values.append(B01_value)

        B02_value = data.sel(x=lon, y=lat, band=2, method="nearest").values
        B02_values.append(B02_value)

        B03_value = data.sel(x=lon, y=lat, band=3, method="nearest").values
        B03_values.append(B03_value)

        B04_value = data.sel(x=lon, y=lat, band=4, method="nearest").values
        B04_values.append(B04_value)

        B05_value = data.sel(x=lon, y=lat, band=5, method="nearest").values
        B05_values.append(B05_value)

        B06_value = data.sel(x=lon, y=lat, band=6, method="nearest").values
        B06_values.append(B06_value)

        B07_value = data.sel(x=lon, y=lat, band=7, method="nearest").values
        B07_values.append(B07_value)

        B08_value = data.sel(x=lon, y=lat, band=8, method="nearest").values
        B08_values.append(B08_value)

        B08A_value = data.sel(x=lon, y=lat, band=9, method="nearest").values
        B08A_values.append(B08A_value)

        B11_value = data.sel(x=lon, y=lat, band=10, method="nearest").values
        B11_values.append(B11_value)

        B12_value = data.sel(x=lon, y=lat, band=11, method="nearest").values
        B12_values.append(B12_value)

        NIR08_value = data.sel(x=lon, y=lat, band=12, method="nearest").values
        NIR08_values.append(NIR08_value)

        Red_value = data.sel(x=lon, y=lat, band=13, method="nearest").values
        Red_values.append(Red_value)

        Green_value = data.sel(x=lon, y=lat, band=14, method="nearest").values
        Green_values.append(Green_value)

        Blue_value = data.sel(x=lon, y=lat, band=15, method="nearest").values
        Blue_values.append(Blue_value)

        SWIR16_value = data.sel(x=lon, y=lat, band=16, method="nearest").values
        SWIR16_values.append(SWIR16_value)

        SWIR22_value = data.sel(x=lon, y=lat, band=17, method="nearest").values
        SWIR22_values.append(SWIR22_value)

        Coastal_value = data.sel(x=lon, y=lat, band=18, method="nearest").values
        Coastal_values.append(Coastal_value)

        LWIR11_value = data.sel(x=lon, y=lat, band=19, method="nearest").values
        LWIR11_values.append(LWIR11_value)
    
    # Create a DataFrame with the band values
    # Create a DataFrame to store the band values
    df = pd.DataFrame()
    df['B01'] = B01_values
    df['B02'] = B02_values
    df['B03'] = B03_values
    df['B04'] = B04_values
    df['B05'] = B05_values
    df['B06'] = B06_values
    df['B07'] = B07_values
    df['B08'] = B08_values
    df['B08A'] = B08A_values
    df['B11'] = B11_values
    df['B12'] = B12_values
    df['NIR08'] = NIR08_values
    df['Red'] = Red_values
    df['Green'] = Green_values
    df['Blue'] = Blue_values
    df['SWIR16'] = SWIR16_values
    df['SWIR22'] = SWIR22_values
    df['Coastal'] = Coastal_values
    df['LWIR11'] = LWIR11_values
    
    return df

# Mapping satellite data with training data.
final_data = map_satellite_data('combined.tiff', 'training_data_uhi_index.csv')

# Display the final data
final_data.head()

def getAverages (lst1, lst2):
    return [(a + b) / 2 for a, b in zip(lst1, lst2)]

# Calculate NDVI (Normalized Difference Vegetation Index) and handle division by zero by replacing infinities with NaN.
final_data['NDVI_S2'] = (final_data['B08'] - final_data['B04']) / (final_data['B08'] + final_data['B04'])
final_data['NDVI_S2'] = final_data['NDVI_S2'].replace([np.inf, -np.inf], np.nan) 

final_data['NDVI_LST'] = (final_data['NIR08'] - final_data['Red']) / (final_data['NIR08'] + final_data['Red'])
final_data['NDVI_LST'] = final_data['NDVI_LST'].replace([np.inf, -np.inf], np.nan)

final_data['NDVI_S2_LST'] = getAverages((final_data['B08'] - final_data['B04']) / (final_data['B08'] + final_data['B04']), ((final_data['NIR08'] - final_data['Red']) / (final_data['NIR08'] + final_data['Red'])))
final_data['NDVI_S2_LST'] = final_data['NDVI_S2_LST'].replace([np.inf, -np.inf], np.nan)
final_data['NDWI_S2'] = (final_data['B03'] - final_data['B08']) / (final_data['B03'] + final_data['B08'])
final_data['NDWI_S2'] = final_data['NDWI_S2'].replace([np.inf, -np.inf], np.nan)
final_data['NDBI_S2'] = (final_data['B11'] - final_data['B08']) / (final_data['B11'] + final_data['B08'])
final_data['NDBI_S2'] = final_data['NDBI_S2'].replace([np.inf, -np.inf], np.nan)


final_data['MSI_LST'] = (final_data['SWIR16'] / final_data['NIR08'])
final_data['MSI_LST'] = final_data['MSI_LST'].replace([np.inf, -np.inf], np.nan)

# Combine two datasets vertically (along columns) using pandas concat function.
def combine_two_datasets(dataset1,dataset2):
    '''
    Returns a  vertically concatenated dataset.
    Attributes:
    dataset1 - Dataset 1 to be combined 
    dataset2 - Dataset 2 to be combined
    '''
    
    data = pd.concat([dataset1,dataset2], axis=1)
    return data

# Combining ground data and final data into a single dataset.
uhi_data = combine_two_datasets(training_df,final_data)
uhi_data.head()

# Remove duplicate rows from the DataFrame based on specified columns and keep the first occurrence
columns_to_check = ['B01','B04','B06','B08','LWIR11']
for col in columns_to_check:
    # Check if the value is a numpy array and has more than one dimension
    uhi_data[col] = uhi_data[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) and x.ndim > 0 else x)

# Now remove duplicates
uhi_data = uhi_data.drop_duplicates(subset=columns_to_check, keep='first')
uhi_data.head()

# Resetting the index of the dataset
uhi_data=uhi_data.reset_index(drop=True)

uhi_data = uhi_data[['B01','B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12','NIR08','Red','Green','Blue','SWIR16','SWIR22','Coastal','LWIR11','NDVI_S2_LST','NDWI_S2','NDBI_S2','MSI_LST','UHI Index']]
uhi_data.head()


X = uhi_data.drop(columns=['UHI Index']).values
y = uhi_data ['UHI Index'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=123)

# Scale the training and test data using standardscaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the Random Forest model on the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

# Make predictions on the training data
insample_predictions = model.predict(X_train)

# calculate R-squared score for in-sample predictions
Y_train = y_train.tolist()
r2_score(Y_train, insample_predictions)

# Make predictions on the test data
outsample_predictions = model.predict(X_test)

# calculate R-squared score for out-sample predictions
Y_test = y_test.tolist()
r2_score(Y_test, outsample_predictions)

#Reading the coordinates for the submission
test_file = pd.read_csv(submission_template)
test_file.head()


val_data = map_satellite_data(combined_tiff, submission_template)

val_data['NDVI'] = (val_data['B08'] - val_data['B04']) / (val_data['B08'] + val_data['B04'])
val_data['NDVI'] = val_data['NDVI'].replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN

val_data['NDVI_S2'] = (val_data['B08'] - val_data['B04']) / (val_data['B08'] + val_data['B04'])
val_data['NDVI_S2'] = val_data['NDVI_S2'].replace([np.inf, -np.inf], np.nan) 

val_data['NDVI_LST'] = (val_data['NIR08'] - val_data['Red']) / (val_data['NIR08'] + val_data['Red'])
val_data['NDVI_LST'] = val_data['NDVI_LST'].replace([np.inf, -np.inf], np.nan)

val_data['NDVI_S2_LST'] = getAverages((val_data['B08'] - val_data['B04']) / (val_data['B08'] + val_data['B04']), ((val_data['NIR08'] - val_data['Red']) / (val_data['NIR08'] + val_data['Red'])))
val_data['NDWI_S2'] = (val_data['B03'] - val_data['B08']) / (val_data['B03'] + val_data['B08'])
val_data['NDBI_S2'] = (val_data['B11'] - val_data['B08']) / (val_data['B11'] + val_data['B08'])

val_data['MSI_LST'] = (val_data['SWIR16'] / val_data['NIR08'])

submission_val_data=val_data.loc[:,['B01','B02','B03','B04','B05','B06','B07','B08','B08A','B11','B12','NIR08','Red','Green','Blue','SWIR16','SWIR22','Coastal','LWIR11','NDVI_S2_LST','NDWI_S2','NDBI_S2','MSI_LST']]
submission_val_data.head()

submission_val_data = submission_val_data.values
transformed_submission_data = sc.transform(submission_val_data)

final_predictions = model.predict(transformed_submission_data)
final_prediction_series = pd.Series(final_predictions)

submission_df = pd.DataFrame({'Longitude':test_file['Longitude'].values, 'Latitude':test_file['Latitude'].values, 'UHI Index':final_prediction_series.values})

#Displaying the submission dataframe
print(submission_df)

submission_df.to_csv("submission.csv",index = False)