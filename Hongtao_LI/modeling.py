import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Preparation
# ======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

# Load data
df1 = pd.read_csv("Branch1.csv")
df2 = pd.read_csv("Branch2.csv")
df3 = pd.read_csv("Branch3.csv")
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.drop('Customer_ID', axis=1)

# ======================
# 2. Logistic Regression (from logistic.py)
# ======================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Logistic preprocessing
df_logit = df.copy()
le = LabelEncoder()
df_logit['Gender'] = le.fit_transform(df_logit['Gender'])

# Handle missing values
imputer = SimpleImputer(strategy='median')
numeric_columns = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use']
df_logit[numeric_columns] = imputer.fit_transform(df_logit[numeric_columns])

X_logit = df_logit.drop('Left', axis=1)
y_logit = df_logit['Left']

X_train_logit, X_test_logit, y_train_logit, y_test_logit = train_test_split(
    X_logit, y_logit, test_size=0.2, random_state=42, stratify=y_logit
)

# Scale features
scaler_logit = StandardScaler()
X_train_logit_scaled = scaler_logit.fit_transform(X_train_logit)
X_test_logit_scaled = scaler_logit.transform(X_test_logit)

# Train model with best parameters from logistic.py
logit_model = LogisticRegression(
    solver='saga',
    penalty='elasticnet',
    C=0.001,
    l1_ratio=0.3,
    class_weight={0:1, 1:4},
    max_iter=5000,
    random_state=42
).fit(X_train_logit_scaled, y_train_logit)

# Initialize results dictionary
results = {}

# ======================
# 3. KNN Implementation (from knn.py)
# ======================
from sklearn.neighbors import KNeighborsClassifier

# KNN preprocessing
df_knn = df.copy()
le = LabelEncoder()
df_knn['Gender'] = le.fit_transform(df_knn['Gender'])

features_knn = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']
X_knn = df_knn[features_knn]
y_knn = df_knn['Left']

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn, y_knn, test_size=0.2, random_state=42
)

knn_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    KNeighborsClassifier(
        n_neighbors=27,           # Updated from knn.py best parameters
        weights='distance',       # Updated from knn.py best parameters
        metric='euclidean',      # Updated from knn.py best parameters
        p=1                      # Updated from knn.py best parameters
    )
)

knn_model = knn_pipeline.fit(X_train_knn, y_train_knn)

# ======================
# 4. XGBoost Implementation (from xgBoost.py)
# ======================
from xgboost import XGBClassifier

# XGBoost preprocessing
df_xgb = df.copy()
df_xgb['Salary'].fillna(df_xgb['Salary'].mean(), inplace=True)
le = LabelEncoder()
df_xgb['Gender'] = le.fit_transform(df_xgb['Gender'])

features_xgb = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']
X_xgb = df_xgb[features_xgb]
y_xgb = df_xgb['Left']

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42
)

xgb_model = XGBClassifier(
    max_depth=3,           # From xgBoost.py best parameters
    learning_rate=0.05,    
    n_estimators=100,      
    subsample=0.8,      
    colsample_bytree=1.0,  
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
).fit(X_train_xgb, y_train_xgb)

# ======================
# 5. Neural Network (from nn.py)
# ======================
from sklearn.neural_network import MLPClassifier

# NN preprocessing
df_nn = df.copy()
df_nn['Salary'] = df_nn['Salary'].fillna(df_nn['Salary'].mean())
df_nn['Tenure'] = df_nn['Tenure'].fillna(df_nn['Tenure'].median())
df_nn['Score'] = df_nn['Score'].fillna(df_nn['Score'].median())
df_nn['Balance'] = df_nn['Balance'].fillna(df_nn['Balance'].median())
df_nn['Products_in_Use'] = df_nn['Products_in_Use'].fillna(df_nn['Products_in_Use'].median())
df_nn['Age'] = df_nn['Age'].fillna(df_nn['Age'].median())

le = LabelEncoder()
df_nn['Gender'] = le.fit_transform(df_nn['Gender'])

features_nn = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']
X_nn = df_nn[features_nn]
y_nn = df_nn['Left']

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(
    X_nn, y_nn, test_size=0.2, random_state=42, stratify=y_nn
)

scaler_nn = StandardScaler()
X_train_nn_scaled = scaler_nn.fit_transform(X_train_nn)
X_test_nn_scaled = scaler_nn.transform(X_test_nn)

nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Updated from best parameters
    activation='tanh',               
    solver='adam',
    alpha=0.1,                        
    learning_rate_init=0.001,
    batch_size=128,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42
).fit(X_train_nn_scaled, y_train_nn)

# ======================
# 6. Model Evaluation
# ======================
from sklearn.metrics import roc_curve

# Dictionary to store test data for each model
test_data = {
    'Logistic': (X_test_logit_scaled, y_test_logit),
    'KNN': (X_test_knn, y_test_knn),
    'XGBoost': (X_test_xgb, y_test_xgb),
    'NeuralNet': (X_test_nn_scaled, y_test_nn)
}

models = {
    'Logistic': logit_model,
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
    if name == 'KNN':
        y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
        y_pred = model.predict(X_test_curr)
    else:
        y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_curr, y_pred)
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test_curr, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'AUC': auc_score,
        'Classification Report': classification_report(y_test_curr, y_pred),
        'Optimal Threshold': optimal_threshold
    }
    
    # Print results for each model
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report:")
    print(results[name]['Classification Report'])

# ======================
# 7. Report Generation
# ======================
def generate_report(results):
    report = {
        'best_model': max(results, key=lambda x: results[x]['AUC']),
        'auc_scores': {k: v['AUC'] for k,v in results.items()},
        'risk_thresholds': {k: v['Optimal Threshold'] for k,v in results.items()},
        'top_features': {
            'Logistic': pd.Series(logit_model.coef_[0], index=X_logit.columns),
            'XGBoost': pd.Series(xgb_model.feature_importances_, index=features_xgb)
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
for model in ['Logistic', 'XGBoost']:
    print(f"\n{model} Feature Importance:")
    print(final_report['top_features'][model].sort_values(ascending=False))