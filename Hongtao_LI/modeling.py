import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Preparation
# ======================
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Load and clean data
df1 = pd.read_csv("Branch1.csv")
df2 = pd.read_csv("Branch2.csv")
df3 = pd.read_csv("Branch3.csv")

# Concatenate the dataframes
df = pd.concat([df1, df2, df3], ignore_index=True)

# Convert 'Gender' to categorical
df['Gender'] = df['Gender'].astype('category')

# Handle missing data
df = df.copy()  # Create a copy to avoid the chained assignment warning
df = df.dropna()  # Drop any remaining NaN values

# Feature engineering
df['Score/Age'] = df['Score'] / df['Age']
df['Products/Balance'] = df['Products_in_Use'] / (df['Balance'] + 1)

# Preprocessing pipeline
numeric_features = ['Score', 'Age', 'Tenure', 'Salary', 'Balance', 
                   'Products_in_Use', 'Score/Age', 'Products/Balance']
categorical_features = ['Gender']

preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split raw data first
X = df.drop(['Customer_ID', 'Left'], axis=1)  # Removed 'Branch' from drop list since we only use Branch1
y = df['Left'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert y to pandas Series
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# ======================
# 2. Logistic Regression
# ======================
from sklearn.linear_model import LogisticRegression

logit_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        class_weight='balanced',
        C=0.1,  # Best value from previous grid search
        solver='liblinear',  # Best performing solver
        max_iter=1000
    ))
])

logit_model = logit_pipe.fit(X_train, y_train)

# ======================
# 3. KNN Implementation
# ======================
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

knn_pipe = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
    ('model', KNeighborsClassifier(
        weights='distance',
        n_neighbors=15,  # Optimal neighbor count
        metric='euclidean'  # Best performing metric
    ))
])

knn_model = knn_pipe.fit(X_train, y_train)

# ======================
# 4. XGBoost Implementation
# ======================
from xgboost import XGBClassifier

xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
        learning_rate=0.1,  # Optimal learning rate
        max_depth=3,  # Best depth from previous search
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

xgb_model = xgb_pipe.fit(X_train, y_train)

# ======================
# 5. Neural Network
# ======================
from sklearn.neural_network import MLPClassifier

nn_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,  # Optimal regularization
        learning_rate_init=0.001,  # Best initial learning rate
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42,
        verbose=0
    ))
])

nn_model = nn_pipe.fit(X_train, y_train)

# ======================
# 6. Model Evaluation
# ======================
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, accuracy_score
import numpy as np

models = {
    'Logistic': logit_model,
    'KNN': knn_model,
    'XGBoost': xgb_model,
    'NeuralNet': nn_model
}

results = {}
print("\nModel Performance Summary:")
print("="*50)

for name, model in models.items():
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'Accuracy': accuracy,
        'AUC': auc_score,
        'Classification Report': classification_report(y_test, y_pred)
    }
    
    # Print results for each model
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print("\nClassification Report:")
    print(results[name]['Classification Report'])

# Calculate optimal thresholds
optimal_thresholds = {}
for name in models:
    y_pred = models[name].predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thresholds[name] = thresholds[optimal_idx]

# ======================
# 7. Business Insights
# ======================
import shap

# Feature importance analysis
explainer_xgb = shap.TreeExplainer(xgb_model.named_steps['model'])
# Convert X_test to numeric before SHAP analysis
X_test_numeric = X_test.copy()
X_test_numeric['Gender'] = X_test_numeric['Gender'].cat.codes
shap_values_xgb = explainer_xgb.shap_values(X_test_numeric)

# ======================
# 8. Visualization
# ======================
import matplotlib.pyplot as plt

# ROC Curves
plt.figure(figsize=(10,6))
for name, model in models.items():
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')

# SHAP Summary Plot
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_xgb, X_test, feature_names=numeric_features + ['Gender_Male'])
plt.title('XGBoost Feature Impact')
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')

# ======================
# 9. Report Generation
# ======================
def generate_report(results):
    report = {
        'best_model': max(results, key=lambda x: results[x]['AUC']),
        'auc_scores': {k: v['AUC'] for k,v in results.items()},
        'risk_thresholds': optimal_thresholds,
        'top_features': {
            'Logistic': pd.Series(logit_model.named_steps['model'].coef_[0],
                                 index=logit_model.named_steps['preprocessor'].get_feature_names_out()),
            'XGBoost': pd.Series(xgb_model.named_steps['model'].feature_importances_,
                                index=xgb_model.named_steps['preprocessor'].get_feature_names_out())
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
