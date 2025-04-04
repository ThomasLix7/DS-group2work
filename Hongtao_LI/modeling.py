# ======================
# 1. Data Preparation
# ======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare dataset
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
# 2. KNN Implementation (from knn.py)
# ======================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Initialize results dictionary
results = {}

knn_model = KNeighborsClassifier(# Updated from knn.py best parameters
    n_neighbors=33,           
    weights='distance',       
    metric='manhattan',      
    p=1                      
).fit(X_train_scaled, y_train)

# ======================
# 3. XGBoost Implementation (from xgBoost.py)
# ======================
from xgboost import XGBClassifier

# Calculate scale_pos_weight
scale_pos_weight = len(y[y==0]) / len(y[y==1])

xgb_model = XGBClassifier( # From xgBoost.py best parameters
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
# 4. Neural Network (from nn.py)
# ======================
from sklearn.neural_network import MLPClassifier

nn_model = MLPClassifier(# Updated from best parameter searching(nn.py)
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
# 5. Model Evaluation
# ======================
from sklearn.metrics import roc_curve

# Dictionary to store test data for each model
test_data = {
    'KNN': (X_test_scaled, y_test),
    'XGBoost': (X_test, y_test), #xgboost model doesn't need scaling
    'NeuralNet': (X_test_scaled, y_test)
}

models = {
    'KNN': knn_model,
    'XGBoost': xgb_model,
    'NeuralNet': nn_model
}

print("\nModel Performance Summary:")
print("="*50)

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
# 6. Report Generation
# ======================
def generate_report(results):
    report = {
        'best_model': max(results, key=lambda x: results[x]['AUC']),
        'auc_scores': {k: v['AUC'] for k,v in results.items()},
        'risk_thresholds': {k: v['Optimal Threshold'] for k,v in results.items()},
        'top_features': {
            'XGBoost': pd.Series(xgb_model.feature_importances_, index=X.columns)
        }
    }
    return report

final_report = generate_report(results)

# Print the final report
print("\nFinal Report:")
print("=============")
print(f"Best Model: {final_report['best_model']}")
print(f"\nAUC Scores:")
for model, auc in final_report['auc_scores'].items():
    print(f"{model}: {auc:.3f}")
print(f"\nRisk Thresholds:")
for model, threshold in final_report['risk_thresholds'].items():
    print(f"{model}: {threshold:.3f}")
print(f"\nTop Features Importance:")
print(f"\nXGBoost Feature Importance:")
print(final_report['top_features']['XGBoost'].sort_values(ascending=False))