# ======================
# Enhanced Modeling with Feature Engineering and Comparison
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

# Load and prepare dataset
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Print column names to verify the target column name
print("\nAvailable columns in the dataset:")
print(df.columns.tolist())

# Try to identify the target column (assuming it's related to customer churn/attrition)
target_column = None
possible_target_names = ['Left', 'Churn', 'Attrition', 'Exited', 'Target']
for col in possible_target_names:
    if col in df.columns:
        target_column = col
        break

if target_column is None:
    raise ValueError("Could not find the target column. Please check the column names in your dataset.")

print(f"\nUsing '{target_column}' as the target column")

# ======================
# 1. Data Preparation
# ======================
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
# 2. Feature Engineering
# ======================
def engineer_features(df):
    """Apply feature engineering to create new features"""
    df = df.copy()
    
    # Create interaction features
    df['Age_Tenure'] = df['Age'] * df['Tenure']
    df['Salary_per_Age'] = df['Salary'] / df['Age']
    df['Balance_per_Age'] = df['Balance'] / df['Age']
    df['Salary_per_Tenure'] = df['Salary'] / (df['Tenure'] + 1)  # Adding 1 to avoid division by zero
    df['Balance_Salary_Ratio'] = df['Balance'] / df['Salary']
    df['Salary_per_Product'] = df['Salary'] / (df['Products_in_Use'] + 1)  # Adding 1 to avoid division by zero
    
    # Create nonlinear transformations
    df['Log_Salary'] = np.log1p(df['Salary'])
    df['Log_Balance'] = np.log1p(df['Balance'])
    df['Salary_squared'] = df['Salary'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    df['Tenure_squared'] = df['Tenure'] ** 2
    
    return df

# ======================
# 3. KNN Implementation
# ======================
knn_model = KNeighborsClassifier(
    n_neighbors=33,           
    weights='distance',       
    metric='manhattan',      
    p=1                      
).fit(X_train_scaled, y_train)

# ======================
# 4. XGBoost Implementation
# ======================
# Calculate scale_pos_weight
scale_pos_weight = len(y[y==0]) / len(y[y==1])

xgb_model = XGBClassifier(
    max_depth=3,          
    learning_rate=0.05,    
    n_estimators=200,      
    subsample=1.0,         
    colsample_bytree=0.8,  
    eval_metric='auc',
    random_state=17,
    scale_pos_weight=scale_pos_weight
).fit(X_train, y_train)

# ======================
# 5. Neural Network Implementation
# ======================
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  
    activation='tanh',               
    solver='adam',
    alpha=0.1,                        
    learning_rate_init=0.01,
    batch_size=256,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=17
).fit(X_train_scaled, y_train)

# ======================
# 6. Model Evaluation
# ======================
# Dictionary to store test data for each model
test_data = {
    'KNN': (X_test_scaled, y_test),
    'XGBoost': (X_test, y_test),  # xgboost model doesn't need scaling
    'NeuralNet': (X_test_scaled, y_test)
}

models = {
    'KNN': knn_model,
    'XGBoost': xgb_model,
    'NeuralNet': nn_model
}

print("\nModel Performance Summary:")
print("="*50)

results = {}
for name, model in models.items():
    # Get corresponding test data
    X_test_curr, y_test_curr = test_data[name]
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_curr, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Handle KNN differently
    if name == 'KNN':
        # For KNN, use its native predict method as it has its own decision boundary logic
        y_pred = model.predict(X_test_curr)
        y_pred_default = y_pred  # KNN's native prediction
        # Also calculate with optimal threshold for comparison
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    else:
        # For other models, use both default and optimal thresholds
        y_pred_default = (y_pred_proba > 0.5).astype(int)
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics using appropriate predictions
    accuracy_default = accuracy_score(y_test_curr, y_pred_default)
    accuracy_optimal = accuracy_score(y_test_curr, y_pred_optimal)
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)  # AUC is independent of threshold
    
    # Store results with optimal threshold metrics
    results[name] = {
        'Accuracy': accuracy_optimal,
        'AUC': auc_score,
        'Classification Report': classification_report(y_test_curr, y_pred_optimal),
        'Optimal Threshold': optimal_threshold
    }
    
    # Print results for each model
    print(f"\n{name} Results:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Accuracy (optimal threshold): {accuracy_optimal:.3f}")
    print(f"Accuracy (default/native): {accuracy_default:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report (using optimal threshold):")
    print(results[name]['Classification Report'])

# ======================
# 7. Feature Engineering Analysis
# ======================
# Prepare engineered features
df_engineered = engineer_features(df)
X_engineered = df_engineered.drop('Left', axis=1)

# Split engineered data
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
    X_engineered, y, 
    test_size=0.2, 
    random_state=17,
    stratify=y
)

# Scale engineered features
X_train_eng_scaled = scaler.fit_transform(X_train_eng)
X_test_eng_scaled = scaler.transform(X_test_eng)

# Train models with engineered features
knn_model_eng = KNeighborsClassifier(
    n_neighbors=33,           
    weights='distance',       
    metric='manhattan',      
    p=1                      
).fit(X_train_eng_scaled, y_train_eng)

xgb_model_eng = XGBClassifier(
    max_depth=3,          
    learning_rate=0.05,    
    n_estimators=200,      
    subsample=1.0,         
    colsample_bytree=0.8,  
    eval_metric='auc',
    random_state=17,
    scale_pos_weight=scale_pos_weight
).fit(X_train_eng, y_train_eng)

nn_model_eng = MLPClassifier(
    hidden_layer_sizes=(64, 32),  
    activation='tanh',               
    solver='adam',
    alpha=0.1,                        
    learning_rate_init=0.01,
    batch_size=256,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=17
).fit(X_train_eng_scaled, y_train_eng)

# Evaluate engineered models
test_data_eng = {
    'KNN': (X_test_eng_scaled, y_test_eng),
    'XGBoost': (X_test_eng, y_test_eng),
    'NeuralNet': (X_test_eng_scaled, y_test_eng)
}

models_eng = {
    'KNN': knn_model_eng,
    'XGBoost': xgb_model_eng,
    'NeuralNet': nn_model_eng
}

print("\nEngineered Features Model Performance Summary:")
print("="*50)

results_eng = {}
for name, model in models_eng.items():
    # Get corresponding test data
    X_test_curr, y_test_curr = test_data_eng[name]
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_curr, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Handle KNN differently
    if name == 'KNN':
        y_pred = model.predict(X_test_curr)
        y_pred_default = y_pred
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    else:
        y_pred_default = (y_pred_proba > 0.5).astype(int)
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy_default = accuracy_score(y_test_curr, y_pred_default)
    accuracy_optimal = accuracy_score(y_test_curr, y_pred_optimal)
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)
    
    # Store results
    results_eng[name] = {
        'Accuracy': accuracy_optimal,
        'AUC': auc_score,
        'Classification Report': classification_report(y_test_curr, y_pred_optimal),
        'Optimal Threshold': optimal_threshold
    }
    
    # Print results
    print(f"\n{name} Results (with engineered features):")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Accuracy (optimal threshold): {accuracy_optimal:.3f}")
    print(f"Accuracy (default/native): {accuracy_default:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report (using optimal threshold):")
    print(results_eng[name]['Classification Report'])

# ======================
# 8. Report Generation
# ======================
def generate_report(results, results_eng):
    report = {
        'best_model': max(results, key=lambda x: results[x]['AUC']),
        'best_model_eng': max(results_eng, key=lambda x: results_eng[x]['AUC']),
        'auc_scores': {k: v['AUC'] for k,v in results.items()},
        'auc_scores_eng': {k: v['AUC'] for k,v in results_eng.items()},
        'risk_thresholds': {k: v['Optimal Threshold'] for k,v in results.items()},
        'risk_thresholds_eng': {k: v['Optimal Threshold'] for k,v in results_eng.items()},
        'top_features': {
            'XGBoost': pd.Series(xgb_model.feature_importances_, index=X.columns)
        },
        'top_features_eng': {
            'XGBoost': pd.Series(xgb_model_eng.feature_importances_, index=X_engineered.columns)
        }
    }
    return report

final_report = generate_report(results, results_eng)

# Print the final report
print("\nFinal Report:")
print("=============")
print(f"Best Model (Original): {final_report['best_model']}")
print(f"Best Model (Engineered): {final_report['best_model_eng']}")
print(f"\nAUC Scores (Original):")
for model, auc in final_report['auc_scores'].items():
    print(f"{model}: {auc:.3f}")
print(f"\nAUC Scores (Engineered):")
for model, auc in final_report['auc_scores_eng'].items():
    print(f"{model}: {auc:.3f}")
print(f"\nRisk Thresholds (Original):")
for model, threshold in final_report['risk_thresholds'].items():
    print(f"{model}: {threshold:.3f}")
print(f"\nRisk Thresholds (Engineered):")
for model, threshold in final_report['risk_thresholds_eng'].items():
    print(f"{model}: {threshold:.3f}")
print(f"\nTop Features Importance (Original):")
print(f"\nXGBoost Feature Importance:")
print(final_report['top_features']['XGBoost'].sort_values(ascending=False))
print(f"\nTop Features Importance (Engineered):")
print(f"\nXGBoost Feature Importance:")
print(final_report['top_features_eng']['XGBoost'].sort_values(ascending=False))
