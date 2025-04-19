import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("data/dataset.csv")

# Handle missing values
data.fillna(data.median(numeric_only=True), inplace=True)  # Fill numeric NaNs with median
data.fillna("Unknown", inplace=True)  # Fill categorical NaNs with "Unknown"

# Handle outliers using IQR method
for column in data.select_dtypes(include=np.number).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

# Encode categorical values
categorical_columns = data.select_dtypes(include="object").columns
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Create new variables if needed
data["BMI"] = data["Weight"] / (data["Height"] / 100) ** 2  # Example: BMI calculation if columns exist

# Exploratory Data Analysis (EDA)
# Distribution of numeric features
for column in data.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Categorical feature distribution
for column in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=data[column])
    plt.title(f"Distribution of {column}")
    plt.xticks(rotation=45)
    plt.show()