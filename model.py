import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

# Confusion matrix for Random Forest
print("\nRandom Forest Confusion Matrix:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_model.classes_)
disp_rf.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()

# logistic regression
print("Logistic Regression Model Results:")
print(classification_report(y_test, y_pred_lr))
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# Confusion matrix for Logistic Regression
print("\nLogistic Regression Confusion Matrix:")
cm_lr = confusion_matrix(y_test, y_pred_lr)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=lr_model.classes_)
disp_lr.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

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

# Confusion matrix for Tuned Random Forest
print("\nTuned Random Forest Confusion Matrix:")
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf_tuned)
disp_rf_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_rf_tuned, display_labels=rf_tuned.classes_)
disp_rf_tuned.plot(cmap="Blues")
plt.title("Tuned Random Forest Confusion Matrix")
plt.show()

# Evaluation of Tuned Logistic Regression
print("\n=== Tuned Logistic Regression Model Results ===")
print(classification_report(y_test, y_pred_lr_tuned))
print("Accuracy:", accuracy_score(y_test, y_pred_lr_tuned))

# Confusion matrix for Tuned Logistic Regression
print("\nTuned Logistic Regression Confusion Matrix:")
cm_lr_tuned = confusion_matrix(y_test, y_pred_lr_tuned)
disp_lr_tuned = ConfusionMatrixDisplay(confusion_matrix=cm_lr_tuned, display_labels=lr_tuned.classes_)
disp_lr_tuned.plot(cmap="Blues")
plt.title("Tuned Logistic Regression Confusion Matrix")
plt.show()

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

# Confusion matrix for Random Forest after SMOTE
print("\nRandom Forest (with SMOTE) Confusion Matrix:")
cm_rf_sm = confusion_matrix(y_test, y_pred_rf_sm)
disp_rf_sm = ConfusionMatrixDisplay(confusion_matrix=cm_rf_sm, display_labels=rf_smote.classes_)
disp_rf_sm.plot(cmap="Blues")
plt.title("Random Forest (with SMOTE) Confusion Matrix")
plt.show()

# Evaluation of Logistic Regression after SMOTE
print("\n=== Logistic Regression Model Results (with SMOTE) ===")
print(classification_report(y_test, y_pred_lr_sm))
print("Accuracy:", accuracy_score(y_test, y_pred_lr_sm))

# Confusion matrix for Logistic Regression after SMOTE
print("\nLogistic Regression (with SMOTE) Confusion Matrix:")
cm_lr_sm = confusion_matrix(y_test, y_pred_lr_sm)
disp_lr_sm = ConfusionMatrixDisplay(confusion_matrix=cm_lr_sm, display_labels=lr_smote.classes_)
disp_lr_sm.plot(cmap="Blues")
plt.title("Logistic Regression (with SMOTE) Confusion Matrix")
plt.show()