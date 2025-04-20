import pandas as pd
import numpy as np


# Load the dataset
data = pd.read_csv("data/dataset.csv")

# look at the dataset
print(data.head())
print(data.info())
print(data.describe()) 

# creating target variable
data['At_Risk'] = data['Recovery_Time'].apply(lambda x: 1 if x > 5 else 0)
# drop the Recovery_Time column
data.drop(columns=['Recovery_Time'], inplace=True)

## DATA PREPROCESSING
