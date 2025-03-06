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
candy_colors = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB", "#9A72C7"]

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

candy_palette = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB", "#9A72C7"]

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
# 6. Models feature weights
# ======================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setting the Seaborn style
sns.set_style("whitegrid")

# colour scheme
candy_palette = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB", "#9A72C7"]

# Creating Submaps
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 3 models, 1 row, 3 columns

# ========== Logistic Regression ==========
logit_weights = pd.Series(logit_model.coef_[0], index=X_logit.columns).sort_values()
ax1 = axes[0]
ax1.barh(logit_weights.index, logit_weights.values, color=candy_palette[:len(logit_weights)])
ax1.set_title("Logistic Regression Feature Importance", fontsize=14, fontweight='bold')
ax1.set_xlabel("Coefficient Value", fontsize=12)
ax1.set_ylabel("Features", fontsize=12)
for i, v in enumerate(logit_weights.values):
    ax1.text(v, i, f'{v:.3f}', ha='left', va='center', fontsize=11, color='dimgray')

# ========== XGBoost ==========
xgb_weights = pd.Series(xgb_model.feature_importances_, index=X_xgb.columns).sort_values()
ax2 = axes[1]
ax2.barh(xgb_weights.index, xgb_weights.values, color=candy_palette[:len(xgb_weights)])
ax2.set_title("XGBoost Feature Importance", fontsize=14, fontweight='bold')
ax2.set_xlabel("Feature Importance Score", fontsize=12)
ax2.set_ylabel("Features", fontsize=12)
for i, v in enumerate(xgb_weights.values):
    ax2.text(v, i, f'{v:.3f}', ha='left', va='center', fontsize=11, color='dimgray')

# ========== Neural Network ==========
nn_weights = pd.Series(np.abs(nn_model.coefs_[0]).sum(axis=1), index=X_nn.columns).sort_values()
ax3 = axes[2]
ax3.barh(nn_weights.index, nn_weights.values, color=candy_palette[:len(nn_weights)])
ax3.set_title("Neural Network Feature Importance", fontsize=14, fontweight='bold')
ax3.set_xlabel("Summed Absolute Weight", fontsize=12)
ax3.set_ylabel("Features", fontsize=12)
for i, v in enumerate(nn_weights.values):
    ax3.text(v, i, f'{v:.3f}', ha='left', va='center', fontsize=11, color='dimgray')

# Restructuring of the layout
plt.tight_layout()
sns.despine()

# Show image
plt.show()

# ========================
# 1. SHAP Summary Plot (Global Importance)
# ========================
# SHAP
# Calculate the SHAP value (using XGBoost as an example)
explainer = shap.Explainer(xgb_model, X_test_xgb)
shap_values = explainer(X_test_xgb)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_xgb)

'''
import shap
import numpy as np
import matplotlib.pyplot as plt

# Compute SHAP interpreter
explainer_logit = shap.Explainer(logit_model, X_test_logit_scaled)
shap_values_logit = explainer_logit(X_test_logit_scaled)

explainer_knn = shap.Explainer(knn_model.predict_proba, X_test_knn)
shap_values_knn = explainer_knn(X_test_knn)

explainer_xgb = shap.Explainer(xgb_model, X_test_xgb)
shap_values_xgb = explainer_xgb(X_test_xgb)

# Fix for MLPClassifier (convert NumPy)
X_test_nn_array = np.array(X_test_nn_scaled)
explainer_nn = shap.Explainer(nn_model.predict_proba, X_test_nn_array)
shap_values_nn = explainer_nn(X_test_nn_array)

# Creating subgraphs: 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ========== Logistic Regression SHAP ==========
plt.sca(axes[0, 0])
shap.summary_plot(shap_values_logit, X_test_logit_scaled, show=False)
axes[0, 0].set_title("Logistic Regression SHAP Summary")

# ========== KNN SHAP ==========
plt.sca(axes[0, 1])
shap.summary_plot(shap_values_knn, X_test_knn, show=False)
axes[0, 1].set_title("KNN SHAP Summary")

# ========== XGBoost SHAP ==========
plt.sca(axes[1, 0])
shap.summary_plot(shap_values_xgb, X_test_xgb, show=False)
axes[1, 0].set_title("XGBoost SHAP Summary")

# ========== Neural Network SHAP ==========
plt.sca(axes[1, 1])
shap.summary_plot(shap_values_nn, X_test_nn_array, show=False)
axes[1, 1].set_title("Neural Network SHAP Summary")

# Adjustment of subgraph layout
plt.tight_layout()
plt.show()
'''

# ========================
# 2. SHAP Bar Plot (Characteristic Importance Bar Plot)
# ========================
sns.set_style("whitegrid")

candy_palette = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB", "#9A72C7"]

plt.figure(figsize=(8, 6))

# Plotting SHAP Feature Importance Histograms
shap.summary_plot(shap_values, X_test_xgb, plot_type="bar", color=candy_palette[1])

plt.title("SHAP Feature Importance", fontsize=14, fontweight='bold')
plt.xlabel("Mean Absolute SHAP Value", fontsize=12)
plt.ylabel("Features", fontsize=12)

sns.despine()

plt.show()

# ========================
# 4. SHAP Dependence Plot (feature dependencies)
# ========================
shap.dependence_plot("Salary", shap_values.values, X_test_xgb)

# ========================
# 5. SHAP Decision Plot
# ========================
shap.decision_plot(explainer.expected_value, shap_values.values[:10], X_test_xgb.iloc[:10])
plt.show()

shap.initjs()
sample_idx = 10

# SHAO Force Plot
force_plot = shap.force_plot(explainer.expected_value, shap_values[sample_idx].values, X_test_xgb.iloc[sample_idx])
# Saving SHAP tries as HTML files
shap.save_html("shap_force_plot.html", force_plot)


# ========================
# 1. Precision-Recall curves
# ========================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Defining colours
colors = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB"]
model_names = ['Logistic', 'KNN', 'XGBoost', 'NeuralNet']

# Creating PR Graphs
plt.figure(figsize=(10, 6))

for i, model_name in enumerate(model_names):
    X_test_curr, y_test_curr = test_data[model_name]
    model = models[model_name]

    # 计算 Precision-Recall
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test_curr, y_pred_proba)
    avg_precision = average_precision_score(y_test_curr, y_pred_proba)

    # plot
    plt.plot(recall, precision, color=colors[i], lw=2,
             label=f"{model_name} (AP={avg_precision:.2f})")

# Image Settings
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend(loc="best")
plt.grid(True)

# Show image
plt.show()

# ========================
# 2. Model calibration assessment
# ========================
from sklearn.calibration import calibration_curve

# colour scheme
colors = ["#AFD3E7", "#FCA3A3", "#ED5F5F", "#FFA74F", "#D0BBDB"]

# Creating calibration graphs
plt.figure(figsize=(10, 6))

for i, model_name in enumerate(['Logistic', 'KNN', 'XGBoost', 'NeuralNet']):
    X_test_curr, y_test_curr = test_data[model_name]
    model = models[model_name]

    # Obtaining Predictive Probabilities
    y_pred_proba = model.predict_proba(X_test_curr)[:, 1]

    # Calculation of calibration curves
    prob_true, prob_pred = calibration_curve(y_test_curr, y_pred_proba, n_bins=10)

    # Plotting calibration curves, using specified colours
    plt.plot(prob_pred, prob_true, marker='o', label=f"{model_name}", color=colors[i])

# Plotting perfectly calibrated lines (y=x lines)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")

# Image Settings
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.title("Model Calibration Plot")
plt.legend(loc="best")
plt.grid(True)

# Show image
plt.show()

# ========================
# 3. Statistical significance testing
# ========================
from scipy.stats import ttest_ind

# Storing the predicted probabilities of different models
model_probs = {}
for model_name in model_names:
    X_test_curr, y_test_curr = test_data[model_name]
    model = models[model_name]
    model_probs[model_name] = model.predict_proba(X_test_curr)[:, 1]

# Statistical significance test (t-test) was performed
stat_results = {}
model_list = list(model_probs.keys())

for i in range(len(model_list)):
    for j in range(i + 1, len(model_list)):
        model1, model2 = model_list[i], model_list[j]
        stat, p_value = ttest_ind(model_probs[model1], model_probs[model2])
        stat_results[f"{model1} vs {model2}"] = p_value

# Convert statistical test results into a DataFrame and display it
import pandas as pd
stat_results_df = pd.DataFrame(stat_results.items(), columns=["Model Comparison", "p-value"])

# Show results
print("Statistical Significance Testing Results:")
print(stat_results_df)

# ========================
# 4. Learning curve
# ========================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Uniform colour scheme
train_color = "#AFD3E7"
test_color = "#ED5F5F"

# Training data and model dictionary
test_data = {
    'Logistic': (X_train_logit_scaled, y_train_logit),
    'KNN': (X_train_knn, y_train_knn),
    'XGBoost': (X_train_xgb, y_train_xgb),
    'NeuralNet': (X_train_nn_scaled, y_train_nn)
}

models = {
    'Logistic': logit_model,
    'KNN': knn_model,
    'XGBoost': xgb_model,
    'NeuralNet': nn_model
}

# Creating Learning Curves
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()  # Flatten the 2x2 subgraphs into a 1-dimensional array.

for i, (model_name, model) in enumerate(models.items()):
    X_train_curr, y_train_curr = test_data[model_name]

    # Calculating the learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_curr, y_train_curr, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 10)
    )

    # Calculate mean and standard deviation
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Training curve (uniform light blue)
    axes[i].plot(train_sizes, train_mean, 'o-', label="Training Score", color=train_color)
    axes[i].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=train_color)

    # Test Curve (uniform red, dashed line)
    axes[i].plot(train_sizes, test_mean, 'o-', label="Cross-validation Score", color=test_color, linestyle="dashed")
    axes[i].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color=test_color)

    # Image Settings
    axes[i].set_title(f"{model_name} Learning Curve")
    axes[i].set_xlabel("Training Samples")
    axes[i].set_ylabel("Accuracy")
    axes[i].legend(loc="best")
    axes[i].grid(True)

# Restructuring of the layout
plt.tight_layout()
plt.show()


