import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

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

# Outlier detection and handling using IQR
def handle_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Cap outliers
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Apply outlier handling to numerical columns
handle_outliers(data, numerical_columns)

# Verify changes after outlier handling
print("After handling outliers:")
print(data[numerical_columns].describe())

# scaling numerical columns 
columns_to_scale = ['Blood_Pressure', 'Heart_Rate', 'Treatment_Duration', 'Lab_Test_Results']
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
print("After scaling:")
print(data[numerical_columns].describe())

# Visualize the distribution of numerical features in a grid layout
num_cols = 2  # Number of columns in the grid
num_rows = (len(numerical_columns) + num_cols - 1) // num_cols  # Calculate rows needed
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 5 * num_rows))
axes = axes.flatten()  # Flatten axes for easy iteration

for i, col in enumerate(numerical_columns):
    sns.histplot(data[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Visualize correlations using a heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

## export preprocessed data for models.py
data.to_csv("data/preprocessed_data.csv", index=False)
print("Preprocessed data saved to data/preprocessed_data.csv")