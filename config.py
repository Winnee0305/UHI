### This file is used for global variable declaration ###


import pandas as pd

training_data = "training_data_uhi_index.csv"

# File to store dataset

sentinel2_tiff = "sentinel2.tiff"

landsat_1_tiff = "landsat_1.tiff"

landsat_2_tiff = "landsat_2.tiff"

submission_template = "submission_template.csv"

## Variables to extract from dataset

# Sentinel 2 Bands
sentinel2_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']

# Landsat 8 Bands
landsat_1_bands = ["blue","green","red","nir08","swir16","swir22","coastal"]

landsat_2_bands = ["lwir11"]

## Read the training data and submission template to declare the global variables

training_df = pd.read_csv(training_data, header=0, sep=',')
training_df.head()
# print("Size of the training data: " , training_df.shape)

submission = pd.read_csv(submission_template)
submission.head()
# print("Size of the submission data: " , submission.shape)

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
# print(f"Area of Interest Bounding Box: {bounds}")

# Define the time window
time_window = "2021-06-01/2021-09-01"

# Define the same dimension value for all GeoTIFF
height = 1122
width = 1281

# # Define the variables to train the model
# train_vars = sentinel2_bands + landsat_1_bands + landsat_2_bands + ['UHI INDEX']
# for x in enumerate(train_vars):
#     train_vars[x[0]] = x[0].upper()

# training_variables = 
# [['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','NIR08','RED','GREEN','BLUE','SWIR16','SWIR22','COASTAL','LWIR11','NDVI_S2','NDVI_LST','NDVI_S2_LST','NDWI_S2','NDBI_S2','MSI_LST','UHI Index']]