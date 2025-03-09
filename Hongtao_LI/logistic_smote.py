import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score
from sklearn.metrics import roc_curve, balanced_accuracy_score
from imblearn.over_sampling import SMOTE

# Function to train and evaluate a basic model with SMOTE
def train_basic_model_smote(X_train_scaled, X_test_scaled, y_train, y_test):
    print("\n" + "="*50)
    print("Training Basic Logistic Regression Model with SMOTE (Default Parameters)")
    print("="*50)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Create a basic logistic regression with default parameters
    basic_model = LogisticRegression(random_state=42)
    
    # Train the model on SMOTE-resampled data
    basic_model.fit(X_train_smote, y_train_smote)
    
    # Get predictions
    y_pred_proba = basic_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    auc = roc_auc_score(y_test, y_pred_proba)
    recall_optimal = recall_score(y_test, y_pred_optimal)
    balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
    
    print(f"\nBasic Model with SMOTE Performance (Optimal threshold = {optimal_threshold:.3f}):")
    print(f"Accuracy: {accuracy_optimal:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_optimal:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall Score: {recall_optimal:.4f}")
    
    print("\nClassification Report (Basic Model with SMOTE, Optimal threshold):")
    print(classification_report(y_test, y_pred_optimal))
    
    return basic_model, auc, recall_optimal, optimal_threshold

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

# Train basic model with SMOTE first to establish baseline
basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model_smote(X_train_scaled, X_test_scaled, y_train, y_test)

# Apply SMOTE for fine-tuned model
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
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
auc_score = roc_auc_score(y_test, y_pred_proba)

# Find optimal threshold
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Make predictions with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate additional metrics
accuracy = accuracy_score(y_test, y_pred_optimal)
recall = recall_score(y_test, y_pred_optimal)
balanced_acc = balanced_accuracy_score(y_test, y_pred_optimal)

# Print results
print("\nTest Set Results:")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"ROC AUC: {auc_score:.4f}")
print(f"Recall: {recall:.4f}")

print(f"\nOptimal Threshold ({optimal_threshold:.4f}):")
print("\nClassification Report with Optimal Threshold:")
print(classification_report(y_test, y_pred_optimal))

# Add comparison of basic vs tuned model
print("\n" + "="*50)
print("BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON (BOTH WITH SMOTE)")
print("="*50)
print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
print(f"{'-'*60}")

# Optimal threshold comparison
print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{auc_score:.4f}{'':<7}{((auc_score-basic_auc)/basic_auc)*100:.2f}%")
print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{recall:.4f}{'':<7}{((recall-basic_recall)/basic_recall)*100:.2f}%")
print(f"\nOptimal thresholds: Basic model: {basic_optimal_threshold:.3f}, Fine-tuned model: {optimal_threshold:.3f}")

print("\nSummary:")
print("This SMOTE-based Logistic Regression model was fine-tuned using ROC AUC scoring")

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