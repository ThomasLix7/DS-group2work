"""
Author: Cui
Create Time: 2025/2/27 11:03
You will never walk along.
"""
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
import shap
import matplotlib.pyplot as plt
import seaborn as sns

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
    class_weight={0: 1, 1: 4},
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
        n_neighbors=27,  # Updated from knn.py best parameters
        weights='distance',  # Updated from knn.py best parameters
        metric='euclidean',  # Updated from knn.py best parameters
        p=1  # Updated from knn.py best parameters
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
    max_depth=3,  # From xgBoost.py best parameters
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
print("=" * 50)

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
        'auc_scores': {k: v['AUC'] for k, v in results.items()},
        'risk_thresholds': {k: v['Optimal Threshold'] for k, v in results.items()},
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


# ======================
# 1. Data Distribution Visualization
# ======================
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Selecting Numeric Characteristics
import matplotlib.pyplot as plt
import seaborn as sns

numeric_columns = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use']

# Calculate the appropriate number of rows and columns
n_cols = 2  # Set the number of diagrams to be placed in each row
n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Calculate the total number of rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))  # Appropriate increase in width
axes = axes.flatten()  # Flattening a 2D array for iteration

for i, col in enumerate(numeric_columns):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

# Remove redundant blank subgraphs
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ======================
# 2. Distribution of target variables
# ======================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Low-saturation candy colors
candy_palette = ["#FADADD", "#B5EAD7"]

plt.figure(figsize=(7, 5))

# Charting the counts
ax = sns.countplot(x=df['Left'], palette=candy_palette)

# Adding numeric labels
for p in ax.patches:
    ax.annotate(f'{p.get_height()}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom',
                fontsize=12, color='dimgray', fontweight='bold')

plt.title("Target Variable Distribution (Left)", fontsize=14, fontweight='bold')
plt.xlabel("Left (0 = Stayed, 1 = Left)", fontsize=12)
plt.ylabel("Count", fontsize=12)

# Remove top and right borders
sns.despine()

plt.show()

# ======================
# 3. ROC curve visualization
# ======================
sns.set_style("whitegrid")

# Low-saturation candy colors
candy_colors = ["#FADADD", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#D5AAFF", "#A2D2FF"]

plt.figure(figsize=(8, 6))

# Plotting ROC curves
for i, (name, model) in enumerate(models.items()):
    X_test_curr, y_test_curr = test_data[name]
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]

    # Calculate FPR, TPR, AUC
    fpr, tpr, _ = roc_curve(y_test_curr, y_pred_proba)
    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.3f})",
             color=candy_colors[i % len(candy_colors)], linewidth=2.5, alpha=0.8)

# Plotting diagonals (stochastic classifiers)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, alpha=0.7)

plt.title("ROC Curves for Different Models", fontsize=14, fontweight='bold')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)

plt.legend(loc="lower right", fontsize=11, frameon=True, bbox_to_anchor=(1, 0))

sns.despine()

plt.show()

# ======================
# 4. Confusion Matrix Visualization
# ======================
fig, axes = plt.subplots(1, len(models), figsize=(15, 4))

for i, (name, model) in enumerate(models.items()):
    X_test_curr, y_test_curr = test_data[name]
    y_pred = model.predict(X_test_curr)

    cm = confusion_matrix(y_test_curr, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix - {name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# ======================
# 5. AUC Score Comparison Histogram
# ======================
sns.set_style("whitegrid")

candy_palette = ["#FADADD", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#D5AAFF", "#A2D2FF"]

# Calculating the AUC score
auc_scores = {name: results[name]['AUC'] for name in models}

plt.figure(figsize=(8, 5))

ax = sns.barplot(x=list(auc_scores.keys()), y=list(auc_scores.values()),
                 palette=candy_palette, width=0.6, alpha=0.9, edgecolor="none")

# Adding numeric labels
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, color='dimgray', fontweight='bold')

plt.title("AUC Score Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Model", fontsize=12)
plt.ylabel("AUC Score", fontsize=12)

plt.ylim(0.5, 1)
sns.despine()
plt.show()

# ======================
# 6. Logistic regression feature weights
# ======================
sns.set_style("whitegrid")

candy_palette = ["#FADADD", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#D5AAFF", "#A2D2FF"]

# Obtain and rank logistic regression coefficients
logit_weights = pd.Series(logit_model.coef_[0], index=X_logit.columns).sort_values()

plt.figure(figsize=(8, 5))

ax = logit_weights.plot(kind='barh', color=candy_palette[:len(logit_weights)],
                        alpha=0.9, edgecolor="gray")

for p in ax.patches:
    plt.text(p.get_width(), p.get_y() + p.get_height()/2,
             f'{p.get_width():.3f}',
             ha='left', va='center', fontsize=11, color='dimgray')

plt.title("Logistic Regression Feature Importance", fontsize=14, fontweight='bold')
plt.xlabel("Coefficient Value", fontsize=12)
plt.ylabel("Features", fontsize=12)

sns.despine()

plt.show()

# SHAP
# Calculate the SHAP value (using XGBoost as an example)
explainer = shap.Explainer(xgb_model, X_test_xgb)
shap_values = explainer(X_test_xgb)

# ========================
# 1. SHAP Summary Plot (Global Importance)
# ========================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_xgb)

# ========================
# 2. SHAP Bar Plot (Characteristic Importance Bar Plot)
# ========================
sns.set_style("whitegrid")

candy_palette = ["#FADADD", "#B5EAD7", "#FFDAC1", "#C7CEEA", "#D5AAFF", "#A2D2FF"]

plt.figure(figsize=(8, 6))

# Plotting SHAP Feature Importance Histograms
shap.summary_plot(shap_values, X_test_xgb, plot_type="bar", color=candy_palette[1])

plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold')
plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
plt.ylabel("Features", fontsize=12)

sns.despine()

plt.show()

# ========================
# 3. SHAP Force Plot (Single Sample Interpretation)
# ========================
shap.initjs()
sample_idx = 10
shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X_test_xgb.iloc[sample_idx])

# ========================
# 4. SHAP Dependence Plot (feature dependencies)
# ========================
shap.dependence_plot("Salary", shap_values.values, X_test_xgb)

# ========================
# 5. SHAP Decision Plot
# ========================
shap.decision_plot(explainer.expected_value, shap_values.values[:10], X_test_xgb.iloc[:10])

plt.show()

# SHAO Force Plot
force_plot = shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X_test_xgb.iloc[sample_idx])
# Saving SHAP tries as HTML files
shap.save_html("shap_force_plot.html", force_plot)
