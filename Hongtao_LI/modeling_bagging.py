# ======================
# 1. Data Preparation
# ======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, recall_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingClassifier  # Add bootstrapping

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

# Apply SMOTE for models that need it
smote = SMOTE(random_state=17)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Initialize dictionaries to store models and results
models = {}
results = {}
models_bagging = {}
results_bagging = {}

# ======================
# 2. Logistic Regression (with class weights, no SMOTE)
# ======================
from sklearn.linear_model import LogisticRegression

# Create and fit regular model
base_logit = LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    C=0.001,
    l1_ratio=0.3,
    class_weight={0:1, 1:4},
    max_iter=5000,
    random_state=17
)
logit_model = base_logit.fit(X_train_scaled, y_train)
models['Logistic'] = logit_model

# Create bagging version
logit_model_bagging = BaggingClassifier(
    estimator=base_logit,
    n_estimators=10,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=17
).fit(X_train_scaled, y_train)
models_bagging['Logistic'] = logit_model_bagging

# ======================
# 3. KNN with SMOTE
# ======================
from sklearn.neighbors import KNeighborsClassifier

# Create base KNN model
base_knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='manhattan',
    p=1
)
knn_model = base_knn.fit(X_train_smote, y_train_smote)
models['KNN'] = knn_model

# Create bagging version
knn_model_bagging = BaggingClassifier(
    estimator=base_knn,
    n_estimators=10,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=17
).fit(X_train_smote, y_train_smote)
models_bagging['KNN'] = knn_model_bagging

# ======================
# 4. XGBoost (no SMOTE, using scale_pos_weight)
# ======================
from xgboost import XGBClassifier

# Calculate scale_pos_weight
scale_pos_weight = len(y[y==0]) / len(y[y==1])

# Create base XGBoost model
base_xgb = XGBClassifier(
    max_depth=4,
    learning_rate=0.01,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='auc',
    random_state=17,
    scale_pos_weight=scale_pos_weight
)
xgb_model = base_xgb.fit(X_train_scaled, y_train)
models['XGBoost'] = xgb_model

# Create bagging version
xgb_model_bagging = BaggingClassifier(
    estimator=base_xgb,
    n_estimators=10,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=17
).fit(X_train_scaled, y_train)
models_bagging['XGBoost'] = xgb_model_bagging

# ======================
# 5. Neural Network with SMOTE
# ======================
from sklearn.neural_network import MLPClassifier

# Create base Neural Network model
base_nn = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='tanh',               
    solver='adam',
    alpha=0.01,                        
    learning_rate_init=0.01,
    batch_size=256,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=17
)
nn_model = base_nn.fit(X_train_smote, y_train_smote)
models['NeuralNet'] = nn_model

# Create bagging version
nn_model_bagging = BaggingClassifier(
    estimator=base_nn,
    n_estimators=10,
    max_samples=1.0,
    bootstrap=True,
    oob_score=True,
    random_state=17
).fit(X_train_smote, y_train_smote)
models_bagging['NeuralNet'] = nn_model_bagging

# ======================
# 6. Model Evaluation
# ======================
from sklearn.metrics import roc_curve

# Dictionary to store test data for each model
test_data = {
    'Logistic': (X_test_scaled, y_test),
    'KNN': (X_test_scaled, y_test),
    'XGBoost': (X_test_scaled, y_test),
    'NeuralNet': (X_test_scaled, y_test)
}

print("\nModel Performance Summary:")
print("="*50)

# Evaluate regular models
print("\nRegular Models:")
print("="*30)

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
    
    # Handle KNN differently as in the original code
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
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)
    recall_default = recall_score(y_test_curr, y_pred_default)
    recall_optimal = recall_score(y_test_curr, y_pred_optimal)
    
    # Store results
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
    print(f"Recall (optimal threshold): {recall_optimal:.3f}")
    print(f"Recall (default/native): {recall_default:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report (using optimal threshold):")
    print(results[name]['Classification Report'])

# Evaluate bagging models
print("\nBagging Models:")
print("="*30)

for name, model in models_bagging.items():
    # Get corresponding test data
    X_test_curr, y_test_curr = test_data[name]
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_curr, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Handle KNN differently as in the original code
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
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)
    recall_default = recall_score(y_test_curr, y_pred_default)
    recall_optimal = recall_score(y_test_curr, y_pred_optimal)
    
    # Store results
    results_bagging[name] = {
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
    print(f"Recall (optimal threshold): {recall_optimal:.3f}")
    print(f"Recall (default/native): {recall_default:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report (using optimal threshold):")
    print(results_bagging[name]['Classification Report'])

# ======================
# 7. Comparison Report
# ======================
def generate_comparison_report(regular_results, bagging_results):
    print("\nModel Comparison Report:")
    print("="*50)
    print("\nAUC Scores Comparison:")
    print("-"*30)
    print(f"{'Model':<15} {'Regular':<10} {'Bagging':<10} {'Difference':<10}")
    print("-"*45)
    
    for model in regular_results.keys():
        reg_auc = regular_results[model]['AUC']
        bag_auc = bagging_results[model]['AUC']
        diff = bag_auc - reg_auc
        print(f"{model:<15} {reg_auc:.3f}     {bag_auc:.3f}     {diff:+.3f}")

    # Find best models
    best_regular = max(regular_results.items(), key=lambda x: x[1]['AUC'])
    best_bagging = max(bagging_results.items(), key=lambda x: x[1]['AUC'])
    
    print("\nBest Models:")
    print(f"Regular: {best_regular[0]} (AUC: {best_regular[1]['AUC']:.3f})")
    print(f"Bagging: {best_bagging[0]} (AUC: {best_bagging[1]['AUC']:.3f})")

# Generate comparison report
generate_comparison_report(results, results_bagging)