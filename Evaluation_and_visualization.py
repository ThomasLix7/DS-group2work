"""
 Author: Yukun Cui
"""
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. Data Preparation
# ======================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, recall_score, precision_score
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone

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

# Initialize dictionaries to store models and results
models = {}
results = {}
cv_results = {}

# ======================
# 2. KNN Implementation
# ======================
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(
    n_neighbors=33,
    weights='distance',
    metric='manhattan',
    p=1
).fit(X_train_scaled, y_train)
models['KNN'] = knn_model

# ======================
# 3. XGBoost Implementation
# ======================
from xgboost import XGBClassifier

# Calculate scale_pos_weight
scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

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
models['XGBoost'] = xgb_model

# ======================
# 4. Neural Network
# ======================
from sklearn.neural_network import MLPClassifier

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
models['NeuralNet'] = nn_model

# ======================
# 5. Cross Validation Evaluation with Threshold Optimization
# ======================
from sklearn.metrics import roc_curve, precision_recall_curve

# Dictionary to store test data for each model
test_data = {
    'KNN': (X_train_scaled, y_train, X_test_scaled, y_test),
    'XGBoost': (X_train, y_train, X_test, y_test),
    'NeuralNet': (X_train_scaled, y_train, X_test_scaled, y_test)
}

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)

print("\nCross-Validation Performance Summary (with Threshold Optimization):")
print("="*60)

for name, model in models.items():
    # Get appropriate data for this model
    X_train_curr, y_train_curr, _, _ = test_data[name]
    
    # Initialize lists to store fold results
    fold_aucs = []
    fold_accuracies = []
    fold_recalls = []
    fold_thresholds = []
    
    # Perform cross-validation with threshold optimization
    for train_idx, val_idx in cv.split(X_train_curr, y_train_curr):
        # Split data into train and validation for this fold
        # Handle differently depending on whether X_train_curr is numpy array or pandas DataFrame
        if isinstance(X_train_curr, np.ndarray):
            X_train_fold = X_train_curr[train_idx]
            X_val_fold = X_train_curr[val_idx]
        else:
            X_train_fold = X_train_curr.iloc[train_idx]
            X_val_fold = X_train_curr.iloc[val_idx]
            
        # Always use iloc for pandas Series
        y_train_fold = y_train_curr.iloc[train_idx]
        y_val_fold = y_train_curr.iloc[val_idx]
        
        # Train the model on this fold's training data
        model_fold = clone(model).fit(X_train_fold, y_train_fold)
        
        # Get probabilities
        y_val_proba = model_fold.predict_proba(X_val_fold)[:, 1]
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        fold_thresholds.append(optimal_threshold)
        
        # Apply optimal threshold
        y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        fold_accuracies.append(accuracy_score(y_val_fold, y_val_pred))
        fold_recalls.append(recall_score(y_val_fold, y_val_pred))
        fold_aucs.append(roc_auc_score(y_val_fold, y_val_proba))
    
    # Store cross-validation results
    cv_results[name] = {
        'AUC': {
            'mean': np.mean(fold_aucs),
            'std': np.std(fold_aucs),
            'values': np.array(fold_aucs)
        },
        'Accuracy': {
            'mean': np.mean(fold_accuracies),
            'std': np.std(fold_accuracies),
            'values': np.array(fold_accuracies)
        },
        'Recall': {
            'mean': np.mean(fold_recalls),
            'std': np.std(fold_recalls),
            'values': np.array(fold_recalls)
        },
        'Threshold': {
            'mean': np.mean(fold_thresholds),
            'std': np.std(fold_thresholds),
            'values': np.array(fold_thresholds)
        }
    }
    
    # Print cross-validation results
    print(f"\n{name} Cross-Validation Results:")
    print(f"AUC: {np.mean(fold_aucs):.3f} ± {np.std(fold_aucs):.3f}")
    print(f"Accuracy (optimized threshold): {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
    print(f"Recall (optimized threshold): {np.mean(fold_recalls):.3f} ± {np.std(fold_recalls):.3f}")
    print(f"Optimal Threshold: {np.mean(fold_thresholds):.3f} ± {np.std(fold_thresholds):.3f}")

# ======================
# 6. Test Set Evaluation
# ======================
print("\nTest Set Performance Summary:")
print("="*60)

for name, model in models.items():
    # Get appropriate test data for this model
    _, _, X_test_curr, y_test_curr = test_data[name]
    
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
    auc_score = roc_auc_score(y_test_curr, y_pred_proba)  # AUC is independent of threshold
    recall_optimal = recall_score(y_test_curr, y_pred_optimal)
    
    # Store results with optimal threshold metrics
    results[name] = {
        'Accuracy': accuracy_optimal,
        'AUC': auc_score,
        'Recall': recall_optimal,
        'Classification Report': classification_report(y_test_curr, y_pred_optimal),
        'Optimal Threshold': optimal_threshold,
        'FPR': fpr,
        'TPR': tpr
    }
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test_curr, y_pred_proba)
    results[name]['PR_Curve'] = {'Precision': precision, 'Recall': recall}
    
    # Print results for each model
    print(f"\n{name} Test Results:")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Accuracy (optimal threshold): {accuracy_optimal:.3f}")
    print(f"Accuracy (default/native): {accuracy_default:.3f}")
    print(f"AUC Score: {auc_score:.3f}")
    print(f"Recall Score: {recall_optimal:.3f}")
    print("\nClassification Report (using optimal threshold):")
    print(results[name]['Classification Report'])

# ======================
# 7. Model Comparison Visualizations
# ======================

# -------- Cross-Validation Results Visualization --------
metrics = ['AUC', 'Accuracy', 'Recall']
model_names = list(models.keys())

# Define colors - use light blue and light orange from earlier files
light_blue = (189/255, 217/255, 247/255)
light_orange = (255/255, 202/255, 167/255)
light_purple = (208/255, 187/255, 219/255)
colors = [light_blue, light_orange, light_purple]

# Create a figure with 3 subplots (one for each metric)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    
    # Extract data
    means = [cv_results[model][metric]['mean'] for model in model_names]
    stds = [cv_results[model][metric]['std'] for model in model_names]
    
    # Create bars
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=stds, capsize=10, width=0.5, color=colors, alpha=0.7)
    
    # Add value labels
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + stds[j] + 0.01,
                f'{means[j]:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_xlabel('')  # Remove the "Models" x-axis label
    ax.set_ylabel(f'{metric} Score')
    ax.set_title(f'Cross-Validation {metric} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set reasonable y-limits
    if metric == 'AUC':
        ax.set_ylim(0.7, 1.0)
    else:
        ax.set_ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig('model_cv_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- ROC and Precision-Recall Curves Combined --------
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ROC Curve on the left
ax1 = axes[0]
for i, (name, model) in enumerate(models.items()):
    fpr = results[name]['FPR']
    tpr = results[name]['TPR']
    auc_score = results[name]['AUC']
    
    ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', 
             color=colors[i], linewidth=2.5, alpha=0.8)

# Add diagonal line for random classifier
ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve Comparison')
ax1.legend(loc='lower right')
ax1.grid(linestyle='--', alpha=0.7)

# Precision-Recall Curve on the right
ax2 = axes[1]
for i, (name, model) in enumerate(models.items()):
    precision = results[name]['PR_Curve']['Precision']
    recall = results[name]['PR_Curve']['Recall']
    
    ax2.plot(recall, precision, label=f'{name}', 
             color=colors[i], linewidth=2.5, alpha=0.8)

ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve Comparison')
ax2.legend(loc='upper right')
ax2.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('roc_pr_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- Test Set Performance Comparison --------
plt.figure(figsize=(12, 8))

# Convert results to a DataFrame for easier plotting
metrics_to_plot = ['AUC', 'Accuracy', 'Recall']
performance_data = {model: [results[model][metric] for metric in metrics_to_plot] 
                   for model in model_names}

# Create a grouped bar chart
x = np.arange(len(metrics_to_plot))
width = 0.25
multiplier = 0

for i, (model, scores) in enumerate(performance_data.items()):
    offset = width * multiplier
    rects = plt.bar(x + offset, scores, width, label=model, color=colors[i], alpha=0.7)
    # Add value labels
    for rect, score in zip(rects, scores):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    multiplier += 1

plt.xlabel('')  # Remove the "Metrics" x-axis label
plt.ylabel('Score')
plt.title('Test Set Performance Comparison')
plt.xticks(x + width, metrics_to_plot)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(model_names))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0.5, 1.0)
plt.savefig('test_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- XGBoost Learning Curve --------
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, X, y, title, ylim=None, cv=5,
                       n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(12, 8))
    
    if ylim is not None:
        plt.ylim(*ylim)
    
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, 
        train_sizes=train_sizes, return_times=False)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid(linestyle='--', alpha=0.7)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color=light_orange, 
             label="Training score", linewidth=2.5, alpha=0.8)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color=light_orange)
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color=light_blue, 
             label="Cross-validation score", linewidth=2.5, alpha=0.8)
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=light_blue)
    
    plt.title(title)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('xgboost_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot learning curve for XGBoost - using full training data instead of test set
# and increase the number of points for a smoother curve
X_train_xgb, y_train_xgb, _, _ = test_data['XGBoost']
plot_learning_curve(xgb_model, X_train_xgb, y_train_xgb, 
                   "XGBoost Learning Curve", 
                   ylim=(0.5, 1.0),
                   cv=5,
                   n_jobs=-1,
                   train_sizes=np.linspace(0.1, 1.0, 10))

# ======================
# 8. XGBoost Detailed Analysis
# ======================
print("\nXGBoost Detailed Analysis:")
print("="*60)

# -------- Feature Importance Visualization --------
# Get feature importance from XGBoost model
feature_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)

plt.figure(figsize=(12, 8))
# Use light_orange for the bars
bars = plt.barh(feature_importance.index, feature_importance.values, color=light_orange, alpha=0.8)

# Add importance values to the bars
for i, (feature, importance) in enumerate(feature_importance.items()):
    plt.text(importance + 0.005, i, f'{importance:.4f}', 
             va='center', fontsize=10, fontweight='bold')

plt.title('XGBoost Feature Importance')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- SHAP Value Plot --------
# Calculate SHAP values for XGBoost model
_, _, X_test_xgb, _ = test_data['XGBoost']
explainer = shap.Explainer(xgb_model, X_test_xgb)
shap_values = explainer(X_test_xgb)

# SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_xgb, show=False)
plt.title('SHAP Feature Impact (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_shap_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# -------- Confusion Matrix for XGBoost --------
from sklearn.metrics import confusion_matrix

# Get predictions using optimal threshold
_, _, X_test_xgb, y_test_xgb = test_data['XGBoost']
y_pred_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
optimal_threshold = results['XGBoost']['Optimal Threshold']
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Create confusion matrix
cm = confusion_matrix(y_test_xgb, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'XGBoost Confusion Matrix (Threshold: {optimal_threshold:.3f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 9. Final Report Generation
# ======================
print("\nFinal Model Comparison Report:")
print("="*60)

# Find the best model based on AUC score
best_model_name = max(results, key=lambda x: results[x]['AUC'])
best_model = models[best_model_name]
best_auc = results[best_model_name]['AUC']

print(f"Best Model: {best_model_name} (AUC: {best_auc:.3f})")
print("\nTest Set Performance Summary:")
for model in model_names:
    auc = results[model]['AUC']
    acc = results[model]['Accuracy']
    rec = results[model]['Recall']
    print(f"{model}: AUC={auc:.3f}, Accuracy={acc:.3f}, Recall={rec:.3f}")

print("\nCross-Validation Performance Summary:")
for model in model_names:
    cv_auc = cv_results[model]['AUC']['mean']
    cv_acc = cv_results[model]['Accuracy']['mean']
    cv_rec = cv_results[model]['Recall']['mean']
    print(f"{model}: AUC={cv_auc:.3f}, Accuracy={cv_acc:.3f}, Recall={cv_rec:.3f}")

print("\nXGBoost Top 10 Features:")
for feature, importance in feature_importance.head(10).items():
    print(f"{feature}: {importance:.4f}")

print("\nAnalysis Complete: All visualizations have been saved.") 