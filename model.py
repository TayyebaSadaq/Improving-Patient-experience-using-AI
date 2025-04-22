import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# MODEL IMPORTS #
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


## import data
data = pd.read_csv('data/preprocessed_data.csv')

# define X and y
# X inputs
# y is the target variable
X = data.drop(['At_Risk'], axis=1)
y = data['At_Risk']

# train test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test) # predict

## Logsitic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test) # predict

## Model Evaluation
# random forest
print("Random Forest Model Results:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# logistic regression
print("Logistic Regression Model Results:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))