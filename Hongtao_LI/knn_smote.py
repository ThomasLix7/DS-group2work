import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, precision_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, precision_recall_curve
from collections import Counter

# Function to evaluate a model with various metrics
def evaluate_model(model, X_test, y_test, model_name="Model"):
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
    recall_optimal = recall_score(y_test, y_pred_optimal)
    precision_optimal = precision_score(y_test, y_pred_optimal)
    f1_optimal = f1_score(y_test, y_pred_optimal)
    
    # Print results
    print(f"\n{model_name} Performance:")
    print("="*50)
    print(f"Default threshold (0.5):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print(f"\nOptimal threshold ({optimal_threshold:.4f}):")
    print(f"Accuracy: {accuracy_optimal:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_optimal:.4f}")
    print(f"Precision: {precision_optimal:.4f}")
    print(f"Recall: {recall_optimal:.4f}")
    print(f"F1 Score: {f1_optimal:.4f}")
    
    print("\nClassification Report (Optimal threshold):")
    print(classification_report(y_test, y_pred_optimal))
    
    # Return metrics for comparison
    return {
        'model': model,
        'auc': auc,
        'recall': recall_optimal,
        'precision': precision_optimal,
        'f1': f1_optimal,
        'threshold': optimal_threshold
    }

# Load and preprocess data
print("\nLoading and preprocessing data...")
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Prepare features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Print original class distribution
print("\nOriginal class distribution:")
print(y.value_counts(normalize=True))

# Split the data - using random_state=42 to match knn.py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Calculate class weights for weighted KNN
class_counts = Counter(y_train)
n_samples = len(y_train)
class_weights = {
    0: n_samples / (2 * class_counts[0]),
    1: n_samples / (2 * class_counts[1])
}
sample_weights = np.array([class_weights[label] for label in y_train])

print("\nClass weights:")
print(class_weights)

# Apply SMOTE to create balanced dataset
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print("\nSMOTE-balanced class distribution:")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# =============================================
# Approach 1: Basic KNN with default parameters
# =============================================
print("\n" + "="*50)
print("Approach 1: Basic KNN with default parameters")
print("="*50)

basic_knn = KNeighborsClassifier()
basic_knn.fit(X_train_scaled, y_train)
basic_results = evaluate_model(basic_knn, X_test_scaled, y_test, "Basic KNN")

# =============================================
# Approach 2: Basic KNN with SMOTE
# =============================================
print("\n" + "="*50)
print("Approach 2: Basic KNN with SMOTE")
print("="*50)

smote_knn = KNeighborsClassifier()
smote_knn.fit(X_train_smote, y_train_smote)
smote_results = evaluate_model(smote_knn, X_test_scaled, y_test, "KNN with SMOTE")

# =============================================
# Approach 3: KNN with sample weights
# =============================================
print("\n" + "="*50)
print("Approach 3: KNN with sample weights")
print("="*50)

# Note: KNeighborsClassifier doesn't directly support sample_weight in fit
# We'll use a different approach - adjusting n_neighbors based on class distribution
weighted_knn = KNeighborsClassifier(weights='distance')  # Use distance weighting instead
weighted_knn.fit(X_train_scaled, y_train)
weighted_results = evaluate_model(weighted_knn, X_test_scaled, y_test, "KNN with distance weights")

# =============================================
# Approach 4: Fine-tuned KNN (without SMOTE)
# =============================================
print("\n" + "="*50)
print("Approach 4: Fine-tuned KNN (without SMOTE)")
print("="*50)

# Define a wider parameter grid matching knn.py
param_grid = {
    'n_neighbors': list(range(1, 32, 2)),  # Odd numbers from 1 to 31
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # For Minkowski distance
}

# Grid search
grid_search = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,
    cv=5,
    scoring='roc_auc',  # Match knn.py scoring metric
    n_jobs=-1,
    verbose=1
)

# Fit on original data (not SMOTE)
grid_search.fit(X_train_scaled, y_train)

# Get best model
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val ROC AUC score: {grid_search.best_score_:.4f}")

# Evaluate best model
tuned_results = evaluate_model(best_model, X_test_scaled, y_test, "Fine-tuned KNN")

# =============================================
# Approach 5: Fine-tuned KNN with SMOTE
# =============================================
print("\n" + "="*50)
print("Approach 5: Fine-tuned KNN with SMOTE")
print("="*50)

# Grid search with SMOTE - using the same wider parameter grid
grid_search_smote = GridSearchCV(
    KNeighborsClassifier(),
    param_grid,  # Same wider parameter grid
    cv=5,
    scoring='roc_auc',  # Match knn.py scoring metric
    n_jobs=-1,
    verbose=1
)

# Fit on SMOTE-balanced data
grid_search_smote.fit(X_train_smote, y_train_smote)

# Get best model
best_model_smote = grid_search_smote.best_estimator_
print(f"\nBest parameters: {grid_search_smote.best_params_}")
print(f"Best cross-val ROC AUC score: {grid_search_smote.best_score_:.4f}")

# Evaluate best model
tuned_smote_results = evaluate_model(best_model_smote, X_test_scaled, y_test, "Fine-tuned KNN with SMOTE")

# =============================================
# Compare all approaches
# =============================================
print("\n" + "="*50)
print("COMPARISON OF ALL APPROACHES")
print("="*50)

# Create comparison table
models = [
    ("Basic KNN", basic_results),
    ("KNN with SMOTE", smote_results),
    ("KNN with distance weights", weighted_results),
    ("Fine-tuned KNN", tuned_results),
    ("Fine-tuned KNN with SMOTE", tuned_smote_results)
]

print(f"{'Model':<25}{'AUC':<10}{'Recall':<10}{'Precision':<10}{'F1 Score':<10}{'Threshold':<10}")
print("-" * 75)

for name, results in models:
    print(f"{name:<25}{results['auc']:.4f}{'':<5}{results['recall']:.4f}{'':<5}{results['precision']:.4f}{'':<5}{results['f1']:.4f}{'':<5}{results['threshold']:.4f}")

# Find the best model based on AUC score instead of F1
best_model_name, best_model_results = max(models, key=lambda x: x[1]['auc'])

print("\nBest model based on AUC score:")
print(f"Model: {best_model_name}")
print(f"AUC: {best_model_results['auc']:.4f}")
print(f"Recall: {best_model_results['recall']:.4f}")
print(f"Precision: {best_model_results['precision']:.4f}")
print(f"F1 Score: {best_model_results['f1']:.4f}")
print(f"Threshold: {best_model_results['threshold']:.4f}")

# Save the best model
import joblib
joblib.dump(best_model_results['model'], 'best_knn_model.joblib')
joblib.dump(scaler, 'knn_scaler.joblib')

print("\nBest model saved as 'best_knn_model.joblib'")
print("Scaler saved as 'knn_scaler.joblib'") 