import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, roc_curve
from sklearn.preprocessing import StandardScaler

# Function to train a basic model
def train_basic_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Training Basic KNN Model (Default Parameters)")
    print("="*50)
    
    # Create a basic KNN with default parameters
    basic_model = KNeighborsClassifier()
    
    # Train the model
    basic_model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = basic_model.predict_proba(X_test)[:, 1]
    
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
    
    print(f"\nBasic Model Performance (Optimal threshold = {optimal_threshold:.3f}):")
    print(f"Accuracy: {accuracy_optimal:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall Score: {recall_optimal:.4f}")
    
    print("\nClassification Report (Basic Model, Optimal threshold):")
    print(classification_report(y_test, y_pred_optimal))
    
    return basic_model, auc, recall_optimal, optimal_threshold

# Load and preprocess data
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Prepare features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train basic model first
basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model(X_train_scaled, X_test_scaled, y_train, y_test)

print("\n" + "="*50)
print("Hyperparameter Tuning Process")
print("="*50)

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
    scoring='roc_auc',
    n_jobs=-1, 
    verbose=1
)

# Fit on scaled data
grid_search.fit(X_train_scaled, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val ROC AUC: {grid_search.best_score_:.4f}")

# Evaluate on test set using scaled data
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

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

# Print results
print(f"\nTest Set Results (Optimal threshold = {optimal_threshold:.3f}):")
print(f"Accuracy: {tuned_accuracy_optimal:.4f}")
print(f"AUC Score: {tuned_auc:.4f}")
print(f"Recall Score: {tuned_recall_optimal:.4f}")

print("\nClassification Report (Optimal threshold):")
print(classification_report(y_test, y_pred_optimal))

# Add comparison of basic vs tuned model
print("\n" + "="*50)
print("BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON")
print("="*50)
print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
print(f"{'-'*60}")

# Optimal threshold comparison
print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{tuned_auc:.4f}{'':<7}{((tuned_auc-basic_auc)/basic_auc)*100:.2f}%")
print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{tuned_recall_optimal:.4f}{'':<7}{((tuned_recall_optimal-basic_recall)/basic_recall)*100:.2f}%")
print(f"\nOptimal thresholds: Basic model: {basic_optimal_threshold:.3f}, Fine-tuned model: {optimal_threshold:.3f}")

# Save the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters for future use:")
print(best_params)