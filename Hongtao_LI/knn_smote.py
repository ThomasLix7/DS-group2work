import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.metrics import roc_curve

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Print SMOTE-balanced class distribution
print("\nSMOTE-balanced class distribution:")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# GridSearchCV 
knn_model = KNeighborsClassifier()

param_grid = {
    'n_neighbors': range(1, 31, 2),  # Test odd numbers 1-29
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # For Minkowski distance
}

grid_search = GridSearchCV(
    knn_model, 
    param_grid, 
    cv=5, 
    scoring='roc_auc',  # Changed to ROC AUC for imbalanced classification
    n_jobs=-1, 
    verbose=1
)

# Fit on SMOTE-balanced data
grid_search.fit(X_train_smote, y_train_smote)

# Get best model and parameters
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val ROC AUC: {grid_search.best_score_:.4f}")

# Get predictions
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# Find optimal threshold using ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Make predictions with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Print comprehensive results
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

# Save the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters for future use:")
print(best_params)

# Optional: Save the best model and scaler
import joblib
joblib.dump(best_model, 'best_knn_smote_model.joblib')
joblib.dump(scaler, 'knn_smote_scaler.joblib') 