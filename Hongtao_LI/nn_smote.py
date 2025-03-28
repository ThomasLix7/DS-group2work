# ======================
# 1. Data Preparation
# ======================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, balanced_accuracy_score, 
    roc_auc_score, roc_curve, recall_score
)
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# ======================
# 2. Model Evaluation Implementation
# ======================
def train_basic_model_smote(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate a basic neural network model with SMOTE using default parameters"""
    print("\n" + "="*50)
    print("Training Basic Neural Network Model with SMOTE (Default Parameters)")
    print("="*50)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # Create a basic neural network with default parameters
    basic_model = MLPClassifier(random_state=17)
    
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

# ======================
# 3. Data Loading and Preprocessing
# ======================
# Load and preprocess data
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Prepare features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=17,
    stratify=y  # Maintain class distribution in splits
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# 4. Model Training and Evaluation
# ======================
# Train basic model with SMOTE first to establish baseline
basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model_smote(X_train_scaled, X_test_scaled, y_train, y_test)

print("\n" + "="*50)
print("Hyperparameter Tuning Process with SMOTE")
print("="*50)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Print class distribution after SMOTE
print("\nClass distribution after SMOTE:")
print(pd.Series(y_res).value_counts(normalize=True))

# Modified parameter grid with more options and simpler architectures
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50,25), (64,32), (100,50)],  # Simpler architectures for better generalization
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # Range of regularization strengths
    'learning_rate_init': [0.0001, 0.001, 0.01],
    'batch_size': [64, 128, 256],
    'max_iter': [1000],  # Maximum iterations for convergence
    'solver': ['adam']  # Adam optimizer for efficient training
}

# Initialize model with minimal configuration
model = MLPClassifier(
    random_state=17,
    early_stopping=True,
    validation_fraction=0.1,  # 10% of training data used for validation
    n_iter_no_change=10
)

# Create and run grid search with increased cross-validation folds
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5),  # 5-fold cross-validation for robust evaluation
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_res, y_res)

# Get best model
best_nn = grid_search.best_estimator_

# Print best parameters and score
print("\nBest parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)

# Calculate cross-validation scores using the best model
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(best_nn, X_res, y_res, cv=cv, scoring='roc_auc')
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Calculate learning curve using the best model
train_sizes, train_scores, val_scores = learning_curve(
    best_nn, X_res, y_res,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=5),
    scoring='roc_auc'
)

# Calculate feature importance
result = permutation_importance(
    best_nn,
    X_test_scaled, 
    y_test,
    n_repeats=10,
    random_state=42
)

# Plot feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
})
importance_df = importance_df.sort_values('Importance', ascending=True)

# Make predictions and evaluate
y_pred_proba = best_nn.predict_proba(X_test_scaled)[:, 1]

# Calculate optimal threshold for tuned model
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

# Calculate metrics with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
tuned_accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
tuned_auc = roc_auc_score(y_test, y_pred_proba)
tuned_recall_optimal = recall_score(y_test, y_pred_optimal)
tuned_balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)

# Print results
print(f"\nTest Set Evaluation (Optimal threshold = {optimal_threshold:.3f}):")
print(f"Accuracy: {tuned_accuracy_optimal:.4f}")
print(f"Balanced Accuracy: {tuned_balanced_acc_optimal:.4f}")
print(f"AUC Score: {tuned_auc:.4f}")
print(f"Recall Score: {tuned_recall_optimal:.4f}")

print("\nClassification Report (Optimal threshold):")
print(classification_report(y_test, y_pred_optimal))

# ======================
# 5. Model Comparison
# ======================
print("\n" + "="*50)
print("BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON (BOTH WITH SMOTE)")
print("="*50)
print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
print(f"{'-'*60}")

# Optimal threshold comparison
print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{tuned_auc:.4f}{'':<7}{((tuned_auc-basic_auc)/basic_auc)*100:.2f}%")
print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{tuned_recall_optimal:.4f}{'':<7}{((tuned_recall_optimal-basic_recall)/basic_recall)*100:.2f}%")
print(f"\nOptimal thresholds: Basic model: {basic_optimal_threshold:.3f}, Fine-tuned model: {optimal_threshold:.3f}")

# Optional: Compare with training performance
y_train_pred = best_nn.predict(X_train_scaled)
print("\nTraining Set Accuracy:", accuracy_score(y_train, y_train_pred))

# Add information about model complexity
print("\nModel Complexity Comparison:")
print(f"Basic model: {basic_model.hidden_layer_sizes}")
print(f"Fine-tuned model: {best_nn.hidden_layer_sizes}")
print(f"Basic model parameters: alpha={basic_model.alpha}, learning_rate={basic_model.learning_rate_init}")
print(f"Fine-tuned model parameters: alpha={best_nn.alpha}, learning_rate={best_nn.learning_rate_init}")

print("\nSummary:")
print("This SMOTE-based Neural Network model was fine-tuned using ROC AUC scoring")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best ROC AUC (CV): {grid_search.best_score_:.4f}")
print(f"Test ROC AUC: {tuned_auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")