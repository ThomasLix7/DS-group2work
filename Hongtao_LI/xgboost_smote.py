import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, recall_score, balanced_accuracy_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE

# Function to train and evaluate a basic model with SMOTE
def train_basic_model_smote(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Training Basic XGBoost Model with SMOTE (Default Parameters)")
    print("="*50)
    
    # Apply SMOTE to training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Create a basic XGBoost with default parameters
    basic_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=17
    )
    
    # Train the model on SMOTE-resampled data
    basic_model.fit(X_train_smote, y_train_smote)
    
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

# Calculate scale_pos_weight
scale_pos_weight = len(y[y==0]) / len(y[y==1])

# Print original class distribution
print("\nOriginal class distribution:")
print(y.value_counts(normalize=True))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train basic model with SMOTE first to establish baseline
basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model_smote(X_train, X_test, y_train, y_test)

# Apply SMOTE for fine-tuned models
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Print SMOTE-balanced class distribution
print("\nSMOTE-balanced class distribution:")
print(pd.Series(y_train_smote).value_counts(normalize=True))

# Train three versions of XGBoost:
# 1. With SMOTE
# 2. With scale_pos_weight
# 3. With both

# 1. XGBoost with SMOTE
param_grid_smote = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

xgb_smote = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=17
)

grid_search_smote = GridSearchCV(
    xgb_smote,
    param_grid_smote,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# 2. XGBoost with scale_pos_weight
xgb_weight = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=17
)

grid_search_weight = GridSearchCV(
    xgb_weight,
    param_grid_smote,  # Same parameters
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# 3. XGBoost with both
xgb_both = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    scale_pos_weight=scale_pos_weight,
    random_state=17
)

grid_search_both = GridSearchCV(
    xgb_both,
    param_grid_smote,  # Same parameters
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# Fit all models
print("\nTraining XGBoost with SMOTE...")
grid_search_smote.fit(X_train_smote, y_train_smote)

print("\nTraining XGBoost with scale_pos_weight...")
grid_search_weight.fit(X_train, y_train)

print("\nTraining XGBoost with both SMOTE and scale_pos_weight...")
grid_search_both.fit(X_train_smote, y_train_smote)

# Get best models
models = {
    'SMOTE': grid_search_smote.best_estimator_,
    'Weight': grid_search_weight.best_estimator_,
    'Both': grid_search_both.best_estimator_
}

# Evaluate all models
print("\nModel Comparison:")
print("="*50)

best_tuned_auc = 0
best_tuned_recall = 0
best_tuned_threshold = 0
best_tuned_name = ""

for name, model in models.items():
    print(f"\n{name} Results:")
    print("-"*20)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    recall = recall_score(y_test, y_pred_optimal)
    
    # Track best model for comparison with basic model
    if auc_score > best_tuned_auc:
        best_tuned_auc = auc_score
        best_tuned_recall = recall
        best_tuned_threshold = optimal_threshold
        best_tuned_name = name
    
    print(f"\nBest Parameters:")
    if name == 'SMOTE':
        print(grid_search_smote.best_params_)
        best_cv_score = grid_search_smote.best_score_
    elif name == 'Weight':
        print(grid_search_weight.best_params_)
        best_cv_score = grid_search_weight.best_score_
    else:
        print(grid_search_both.best_params_)
        best_cv_score = grid_search_both.best_score_
    
    print(f"\nModel Performance:")
    print(f"Best ROC AUC (CV): {best_cv_score:.4f}")
    print(f"Test ROC AUC: {auc_score:.4f}")
    print(f"Test Recall: {recall:.4f}")
    
    print(f"\nOptimal Threshold ({optimal_threshold:.4f}):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
    print("\nClassification Report with Optimal Threshold:")
    print(classification_report(y_test, y_pred_optimal))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importance:")
    print(importance_df.head(10))

# Add comparison of basic vs best tuned model
print("\n" + "="*50)
print(f"BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON (BOTH WITH SMOTE)")
print("="*50)
print(f"Best fine-tuned model: {best_tuned_name}")
print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
print(f"{'-'*60}")

# Optimal threshold comparison
print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{best_tuned_auc:.4f}{'':<7}{((best_tuned_auc-basic_auc)/basic_auc)*100:.2f}%")
print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{best_tuned_recall:.4f}{'':<7}{((best_tuned_recall-basic_recall)/basic_recall)*100:.2f}%")
print(f"\nOptimal thresholds: Basic model: {basic_optimal_threshold:.3f}, Fine-tuned model: {best_tuned_threshold:.3f}")

# Save the best model (choose based on results)
best_scores = {
    name: roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    for name, model in models.items()
}

best_method = max(best_scores.items(), key=lambda x: x[1])[0]
best_model = models[best_method]

print(f"\nBest Method: {best_method}")
print(f"Best ROC AUC: {best_scores[best_method]:.4f}")

# Save the best model
import joblib
joblib.dump(best_model, 'best_xgboost_model.joblib') 