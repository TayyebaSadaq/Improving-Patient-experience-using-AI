import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("data/dataset.csv")

# look at the dataset
# print(data.head())
# print(data.info())
# print(data.describe()) 

# creating target variable
data['At_Risk'] = data['Recovery_Time'].apply(lambda x: 1 if x > 5 else 0)
# drop the Recovery_Time column
data.drop(columns=['Recovery_Time'], inplace=True)
# drop the ID column, doctor name, and hospital name columns
data.drop(columns=['Patient_ID', 'Hospital_Name', 'Doctor_Name' ], inplace=True)

# Encode categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
# print(data.info())

## DATA PREPROCESSING
# check for missing values
# print(data.isnull().sum())

# Identify numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
print(data[numerical_columns].describe())
# scaling numerical columns 
columns_to_scale = ['Blood_Pressure', 'Heart_Rate', 'Treatment_Duration', 'Lab_Test_Results']
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
print("After scaling:")
print(data[numerical_columns].describe())
