# ======================
# Enhanced Modeling with Feature Engineering and Comparison
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, recall_score, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# ======================
# 1. Feature Engineering
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
# 2. Model Training Function
# ======================
def train_models(X_train, X_test, y_train, y_test):
    """Train models with minimal output"""
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for models that need it
    smote = SMOTE(random_state=17)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    # KNN with SMOTE
    knn_model = KNeighborsClassifier(
        n_neighbors=7,
        weights='distance',
        metric='manhattan',
        p=1
    ).fit(X_train_smote, y_train_smote)
    
    # XGBoost (no SMOTE, using scale_pos_weight)
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    xgb_model = XGBClassifier(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        subsample=1.0,
        colsample_bytree=0.8,
        eval_metric='auc',
        random_state=17,
        scale_pos_weight=scale_pos_weight
    ).fit(X_train_scaled, y_train)
    
    # Neural Network with SMOTE
    nn_model = MLPClassifier(
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
    ).fit(X_train_smote, y_train_smote)
    
    # Store models
    models = {
        'KNN': knn_model,
        'XGBoost': xgb_model,
        'NeuralNet': nn_model
    }
    
    return models, X_test_scaled, y_test

# ======================
# 3. Model Evaluation Function
# ======================
def evaluate_models(models, X_test, y_test):
    """Evaluate models and return performance metrics"""
    results = {}
    
    for name, model in models.items():
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate optimal threshold
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Handle KNN differently as in the original code
        if name == 'KNN':
            # For KNN, use its native predict method as it has its own decision boundary logic
            y_pred = model.predict(X_test)
            y_pred_default = y_pred  # KNN's native prediction
            # Also calculate with optimal threshold for comparison
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        else:
            # For other models, use both default and optimal thresholds
            y_pred_default = (y_pred_proba > 0.5).astype(int)
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics using appropriate predictions
        accuracy_default = accuracy_score(y_test, y_pred_default)
        accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
        auc_score = roc_auc_score(y_test, y_pred_proba)  # AUC is independent of threshold
        
        # Calculate recall scores by class
        recall_0_default = recall_score(y_test, y_pred_default, pos_label=0)
        recall_1_default = recall_score(y_test, y_pred_default, pos_label=1)
        recall_0_optimal = recall_score(y_test, y_pred_optimal, pos_label=0)
        recall_1_optimal = recall_score(y_test, y_pred_optimal, pos_label=1)
        
        # Store results with optimal threshold metrics
        results[name] = {
            'Accuracy': accuracy_optimal,
            'AUC': auc_score,
            'Recall_Class0': recall_0_optimal,
            'Recall_Class1': recall_1_optimal,
            'Classification Report': classification_report(y_test, y_pred_optimal),
            'Optimal Threshold': optimal_threshold,
            'Default_Accuracy': accuracy_default,
            'Default_Recall_Class0': recall_0_default,
            'Default_Recall_Class1': recall_1_default
        }
    
    return results

# ======================
# 4. Compare and Visualize Function
# ======================
def compare_and_visualize(original_results, engineered_results, model_names):
    """Compare and visualize the performance between original and engineered features"""
    # Create comparison DataFrame
    comparison_data = []
    
    for model in model_names:
        original_auc = original_results[model]['AUC']
        engineered_auc = engineered_results[model]['AUC']
        auc_diff = engineered_auc - original_auc
        
        original_threshold = original_results[model].get('Optimal Threshold', 0.5)
        engineered_threshold = engineered_results[model].get('Optimal Threshold', 0.5)
        
        comparison_data.append({
            'Model': model,
            'Original AUC': original_auc,
            'Engineered AUC': engineered_auc,
            'AUC Difference': auc_diff,
            'Original Threshold': original_threshold,
            'Engineered Threshold': engineered_threshold
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nFeature Engineering Performance Comparison:")
    print("="*80)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Define the specific colors requested
    original_color = (189/255, 217/255, 247/255)  # Light blue (RGB: 189, 217, 247)
    engineered_color = (255/255, 202/255, 167/255)  # Light orange/peach (RGB: 255, 202, 167)
    
    # Visualize the comparison
    plt.figure(figsize=(12, 6))  # Adjusted height for single subplot
    
    # Create AUC comparison bars
    width = 0.35
    x = np.arange(len(model_names))
    
    # Create AUC comparison bars with specified colors
    plt.bar(x - width/2, comparison_df['Original AUC'], width, label='Original Features', alpha=0.9, color=original_color)
    plt.bar(x + width/2, comparison_df['Engineered AUC'], width, label='Engineered Features', alpha=0.9, color=engineered_color)
    
    # Add labels
    plt.xlabel('Models')
    plt.ylabel('AUC Score')
    plt.title('Impact of Feature Engineering on AUC Performance')
    plt.xticks(x, model_names)
    
    # Position the legend inside the plot area instead of outside
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('feature_engineering_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Add a note about thresholds being used
    print("\nNote: All metrics are calculated using optimal thresholds")
    for model in model_names:
        print(f"{model} - Original threshold: {comparison_df.loc[comparison_df['Model']==model, 'Original Threshold'].values[0]:.3f}, " +
              f"Engineered threshold: {comparison_df.loc[comparison_df['Model']==model, 'Engineered Threshold'].values[0]:.3f}")
    
    return comparison_df

# ======================
# 5. Visualize Feature Engineering
# ======================
def visualize_features(engineered_models, X_eng_columns, original_columns):
    """Visualize the XGBoost feature importance after engineering"""
    # Get feature importance from XGBoost model
    xgb_importance = pd.Series(engineered_models['XGBoost'].feature_importances_, index=X_eng_columns)
    xgb_importance = xgb_importance.sort_values(ascending=False)
    
    # Create figure for XGBoost feature importance
    plt.figure(figsize=(14, 10))
    
    # Show top features
    
    bars = plt.barh(range(len(xgb_importance)), xgb_importance.values, color=(189/255, 217/255, 247/255))  # Light blue for original features
    plt.yticks(range(len(xgb_importance)), xgb_importance.index)
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.title('XGBoost Feature Importance (After Engineering)')
    plt.xlabel('Importance Score')
    
    # Highlight engineered features with a different color
    # Using a definitive list of original columns to avoid confusion
    for i, feature in enumerate(xgb_importance.index):
        if feature not in original_columns:
            bars[i].set_color((255/255, 202/255, 167/255))  # Light orange/peach for engineered features
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(189/255, 217/255, 247/255), label='Original Features'),
        Patch(facecolor=(255/255, 202/255, 167/255), label='Engineered Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')
    plt.show()

    return xgb_importance

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    # Load the preprocessed dataset
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
        stratify=y
    )
    
    # Train and evaluate models with original features
    original_models, X_test_scaled, y_test = train_models(X_train, X_test, y_train, y_test)
    original_results = evaluate_models(original_models, X_test_scaled, y_test)
    
    # Apply feature engineering
    df_engineered = engineer_features(df)
    
    # Prepare engineered features and target
    X_eng = df_engineered.drop('Left', axis=1)
    y_eng = df_engineered['Left']
    
    # Split engineered data using the same random state for fair comparison
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
        X_eng, y_eng, 
        test_size=0.2, 
        random_state=17,
        stratify=y_eng
    )
    
    # Train and evaluate models with engineered features
    engineered_models, X_test_eng_scaled, y_test_eng = train_models(X_train_eng, X_test_eng, y_train_eng, y_test_eng)
    engineered_results = evaluate_models(engineered_models, X_test_eng_scaled, y_test_eng)
    
    # Visualize the feature importance after engineering
    feature_importance = visualize_features(engineered_models, X_eng.columns, X.columns)
    
    # Compare results
    comparison = compare_and_visualize(original_results, engineered_results, ['KNN', 'XGBoost', 'NeuralNet'])
