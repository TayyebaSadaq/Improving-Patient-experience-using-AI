import pandas as pd
import numpy as np


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
categorical_columns = data.select_dtypes(include=['object']).
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
print(data.info())

## DATA PREPROCESSING
