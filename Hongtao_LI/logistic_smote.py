import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE

# Load and preprocess data
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Prepare features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Print original class distribution
print("\nOriginal class distribution:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Print SMOTE-balanced class distribution
print("\nSMOTE-balanced class distribution:")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'solver': ['saga'],
    'l1_ratio': [0.2, 0.5, 0.8],
    'max_iter': [1000]
}

# Initialize model
model = LogisticRegression(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit on SMOTE-balanced data
grid_search.fit(X_train_smote, y_train_smote)

# Get best model
best_model = grid_search.best_estimator_

# Print best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)
print(f"Best cross-validation ROC AUC: {grid_search.best_score_:.4f}")

# Make predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Make predictions with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Print results
print("\nTest Set Results:")
print("="*50)
print("\nDefault Threshold (0.5):")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nOptimal Threshold ({optimal_threshold:.4f}):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_model.coef_[0]
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Coefficient', ascending=False))

# Save models
import joblib
joblib.dump(best_model, 'best_logistic_smote_model.joblib')
joblib.dump(scaler, 'logistic_smote_scaler.joblib') 