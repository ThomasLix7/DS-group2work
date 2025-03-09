import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, roc_curve

# Function to train a basic model with default parameters
def train_basic_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Training Basic XGBoost Model (Default Parameters)")
    print("="*50)
    
    # Create a basic XGBoost with default parameters
    basic_model = xgb.XGBClassifier(random_state=42)
    
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

# Train basic model first to establish baseline
basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model(X_train, X_test, y_train, y_test)

print("\n" + "="*50)
print("Hyperparameter Tuning Process")
print("="*50)

# Train XGBoost model with Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get best model
model = grid_search.best_estimator_

# Print grid search results
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV ROC AUC: {:.4f}".format(grid_search.best_score_))

# Make predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]

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

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))