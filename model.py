import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# MODEL IMPORTS #
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


## import data
data = pd.read_csv('data/preprocessed_data.csv')

# --------------------------- BASELINE MODELS ---------------------------

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

# --------------------------- IMPROVEMENTS: CROSS VALIDATION ---------------------------

from sklearn.model_selection import cross_val_score

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("\n=== Random Forest Cross-Validation Accuracy ===")
print(np.mean(rf_cv_scores))

# Cross-validation for Logistic Regression
lr_cv_scores = cross_val_score(lr_model, X, y, cv=5)
print("\n=== Logistic Regression Cross-Validation Accuracy ===")
print(np.mean(lr_cv_scores))


# --------------------------- IMPROVEMENTS: HYPERPARAMETER TUNING ---------------------------

# Tuned Random Forest
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_tuned.fit(X_train, y_train)
y_pred_rf_tuned = rf_tuned.predict(X_test)

# Tuned Logistic Regression
lr_tuned = LogisticRegression(
    C=0.5,       # Tuning regularization strength
    penalty='l2', # Keep standard L2 regularization
    max_iter=1000,
    random_state=42
)
lr_tuned.fit(X_train, y_train)
y_pred_lr_tuned = lr_tuned.predict(X_test)

# Evaluation of Tuned Random Forest
print("\n=== Tuned Random Forest Model Results ===")
print(classification_report(y_test, y_pred_rf_tuned))
print("Accuracy:", accuracy_score(y_test, y_pred_rf_tuned))

# Evaluation of Tuned Logistic Regression
print("\n=== Tuned Logistic Regression Model Results ===")
print(classification_report(y_test, y_pred_lr_tuned))
print("Accuracy:", accuracy_score(y_test, y_pred_lr_tuned))

# --------------------------- IMPROVEMENTS: SMOTE ---------------------------

from imblearn.over_sampling import SMOTE

# Apply SMOTE to training data
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

# Random Forest after SMOTE
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_sm, y_train_sm)
y_pred_rf_sm = rf_smote.predict(X_test)

# Logistic Regression after SMOTE
lr_smote = LogisticRegression(max_iter=1000, random_state=42)
lr_smote.fit(X_train_sm, y_train_sm)
y_pred_lr_sm = lr_smote.predict(X_test)

# Evaluation of Random Forest after SMOTE
print("\n=== Random Forest Model Results (with SMOTE) ===")
print(classification_report(y_test, y_pred_rf_sm))
print("Accuracy:", accuracy_score(y_test, y_pred_rf_sm))

# Evaluation of Logistic Regression after SMOTE
print("\n=== Logistic Regression Model Results (with SMOTE) ===")
print(classification_report(y_test, y_pred_lr_sm))
print("Accuracy:", accuracy_score(y_test, y_pred_lr_sm))