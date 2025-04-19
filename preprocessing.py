import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

## Importing dataset needed for the project
labels = pd.read_csv("data/dataset_labels.csv")
gps_data = pd.read_csv("data/dataset_gps.csv")
mpu_left = pd.read_csv("data/dataset_mpu_left.csv")
mpu_right = pd.read_csv("data/dataset_mpu_right.csv")
gps_mpu_left = pd.read_csv("data/dataset_gps_mpu_left.csv")
gps_mpu_right = pd.read_csv("data/dataset_gps_mpu_right.csv")

## Analysing dataset
print("Labels DataFrame Info:")
print(labels.head())
print("\nGPS DataFrame Info:")
print(gps_data.head())
print("\nMPU Left DataFrame Info:")
print(mpu_left.head())
print("\nMPU Right DataFrame Info:")
print(mpu_right.head())
print("\nGPS MPU Left DataFrame Info:")
print(gps_mpu_left.head())
print("\nGPS MPU Right DataFrame Info:")
print(gps_mpu_right.head())


## Check for missing values
print("GPS data :")
print(gps_data.isnull().sum())
print("\nMPU data left:")
print(mpu_left.isnull().sum())
print("\nMPU data right:")
print(mpu_right.isnull().sum())
print("\nCombined GPS + MPU data left:")
print(gps_mpu_left.isnull().sum())
print("\nCombined GPS + MPU data right:")
print(gps_mpu_right.isnull().sum())

## Handling missing values
print("Handling missing values...")
# drop columns that are entirely null
gps_data.drop(columns=['ageofdgpsdata', 'dgpsid', 'activity', 'annotation'], inplace=True)
# verify if the columns are dropped
print("Columns after dropping:")
print(gps_data.isnull().sum())

## Assessing quality of data
print("GPS DATA: \n ", gps_data.describe())
print("----------")
print("\n MPU DATA LEFT: \n ", mpu_left.describe())
print("----------")
print("\n MPU DATA RIGHT: \n ", mpu_right.describe())
print("----------")
print("\n COMBINED GPS AND MPU DATA LEFT: \n ", gps_mpu_left.describe())
print("----------")
print("\n COMBINED GPS AND MPU DATA LEFT: \n ", gps_mpu_right.describe())
