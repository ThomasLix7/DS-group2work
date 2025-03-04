import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE

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

# Apply SMOTE
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
    
    print(f"\nBest Parameters:")
    if name == 'SMOTE':
        print(grid_search_smote.best_params_)
    elif name == 'Weight':
        print(grid_search_weight.best_params_)
    else:
        print(grid_search_both.best_params_)
    
    print(f"\nDefault Threshold (0.5):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
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