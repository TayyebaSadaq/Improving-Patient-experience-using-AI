import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = r"c:\Users\tayye\Desktop\Improving-Patient-Experience-Using-AI\data\dataset.csv"
data = pd.read_csv(file_path)

# Data Cleaning
# Remove duplicate rows
data.drop_duplicates(inplace=True)

# Handle missing values
# Replace missing numerical values with the median
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

# Replace missing categorical values with the mode
categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Handle noisy data
# Remove outliers using the IQR method for numerical columns
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Normalize data
scaler = MinMaxScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Exploratory Data Analysis (EDA)
# Visualize the distribution of patient satisfaction
plt.figure(figsize=(8, 6))
sns.histplot(data['Patient_Satisfaction'], kde=True, bins=20, color='blue')
plt.title('Distribution of Patient Satisfaction')
plt.xlabel('Patient Satisfaction')
plt.ylabel('Frequency')
plt.show()

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Visualize the count of diagnoses
plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='Diagnosis', order=data['Diagnosis'].value_counts().index, palette='viridis')
plt.title('Count of Diagnoses')
plt.xlabel('Count')
plt.ylabel('Diagnosis')
plt.show()

# Save the cleaned and preprocessed data
cleaned_file_path = r"c:\Users\tayye\Desktop\Improving-Patient-Experience-Using-AI\data\cleaned_dataset.csv"
data.to_csv(cleaned_file_path, index=False)
