import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, roc_auc_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("QM_pre-process/output.csv")
    df = df.drop(['Customer_ID', 'Source'], axis=1)
    
    return df

# New function to train and evaluate a basic model with default parameters
def train_basic_model(X_train, X_test, y_train, y_test):
    # Create a basic logistic regression with default parameters
    basic_model = LogisticRegression(random_state=42)
    
    # Train the model
    basic_model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = basic_model.predict_proba(X_test)[:, 1]
    
    # Calculate optimal threshold
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    auc = roc_auc_score(y_test, y_pred_proba)
    recall_optimal = recall_score(y_test, y_pred_optimal)
    balanced_acc_optimal = balanced_accuracy_score(y_test, y_pred_optimal)
    
    print(f"\nBasic Model Performance (Optimal threshold = {optimal_threshold:.3f}):")
    print(f"Accuracy: {accuracy_optimal:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall Score: {recall_optimal:.4f}")
    
    return basic_model, auc, recall_optimal, optimal_threshold

def train_model(df):
    # Separate features and target
    X = df.drop('Left', axis=1)
    y = df['Left']
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train basic model first to establish baseline
    basic_model, basic_auc, basic_recall, basic_optimal_threshold = train_basic_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Define different solver-penalty combinations to search for the best model parameters
    solvers_penalties = [
        ('lbfgs', 'l2'),     # Efficient for large datasets
        ('newton-cg', 'l2'), # Good for multinomial loss
        ('liblinear', 'l1'), # Good for sparse data
        ('liblinear', 'l2'), # Good for binary classification
        ('saga', 'elasticnet'), # Can handle all penalties
        ('saga', 'l1'),      # Good for sparse data
        ('saga', 'l2'),      # Standard regularization
        ('sag', 'l2')        # Faster for large datasets
    ]
    
    best_score = 0
    best_model = None
    best_params = None
    best_solver_penalty = None
    
    # Try each solver-penalty combination
    for solver, penalty in solvers_penalties:
        # Define base model
        log_model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            max_iter=5000,  # Increased to ensure convergence
            random_state=42
        )
        
        # Define parameter grid based on solver-penalty combination
        if penalty == 'elasticnet':
            param_grid = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}]
            }
        else:
            param_grid = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}, {0:1, 1:5}]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=log_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='roc_auc',  # Primary optimization metric is AUC
            n_jobs=-1
        )
        
        try:
            # Fit and evaluate
            grid_search.fit(X_train_scaled, y_train)
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_solver_penalty = (solver, penalty)
        except Exception as e:
            continue
    
    # Make predictions with best model
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold using ROC curve (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_roc = thresholds[optimal_idx]
    
    # Calculate metrics with optimal threshold
    y_pred_roc = (y_pred_proba >= optimal_threshold_roc).astype(int)
    
    # Calculate metrics
    tuned_auc = roc_auc_score(y_test, y_pred_proba)
    
    # ROC threshold metrics
    accuracy_roc = accuracy_score(y_test, y_pred_roc)
    recall_roc = recall_score(y_test, y_pred_roc)
    
    print("\nFine-tuned Model Performance Summary:")
    print(f"Best solver-penalty: {best_solver_penalty}")
    print(f"Best parameters: {best_params}")
    print(f"Optimal threshold: {optimal_threshold_roc:.3f}")
    print(f"Accuracy: {accuracy_roc:.4f}")
    print(f"AUC: {tuned_auc:.4f}")
    print(f"Recall: {recall_roc:.4f}")
    
    # Choose the threshold that gives the best balanced performance
    optimal_threshold = optimal_threshold_roc
    tuned_recall = recall_roc
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_[0]),
        'coefficient': best_model.coef_[0]  # Actual coefficient values
    }).sort_values('importance', ascending=False)
    
    # Add comparison of basic vs tuned model
    print("\n" + "="*50)
    print("BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
    print(f"{'-'*60}")
    print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{tuned_auc:.4f}{'':<7}{((tuned_auc-basic_auc)/basic_auc)*100:.2f}%")
    print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{tuned_recall:.4f}{'':<7}{((tuned_recall-basic_recall)/basic_recall)*100:.2f}%")
    print(f"\nOptimal thresholds: Basic: {basic_optimal_threshold:.3f}, Fine-tuned: {optimal_threshold:.3f}")
    
    # Print model details for interpretation
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION MODEL INTERPRETATION")
    print("="*50)
    
    # Print intercept
    print(f"Intercept: {best_model.intercept_[0]:.4f}")
    
    # Print coefficients with feature names
    print("\nCoefficients (sorted by absolute value):")
    for idx, row in feature_importance.head(15).iterrows():
        # Determine the effect (positive or negative influence)
        effect = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"{row['feature']:<25}: {row['coefficient']:+.4f} - {effect} probability of leaving")
    
    # Create visualizations
    # 1. Coefficient plot
    plt.figure(figsize=(12, 8))
    plt.title('Logistic Regression Coefficients', fontsize=16)
    
    # Get top N most important features for clearer visualization
    top_n = 15
    top_features = feature_importance.head(top_n)
    
    # Create a new column for color coding
    top_features['effect'] = ['Negative' if coef < 0 else 'Positive' for coef in top_features['coefficient']]
    
    # Use the user specified RGB colors
    # Convert RGB to hex format or normalized RGB tuples
    light_blue = (189/255, 217/255, 247/255)  # Light blue
    light_orange = (255/255, 202/255, 167/255)  # Light orange/peach
    
    # Use the specified colors
    color_palette = {"Positive": light_orange, "Negative": light_blue}
    
    # Plot with the new color scheme
    ax = sns.barplot(x='coefficient', y='feature', data=top_features, hue='effect', palette=color_palette, dodge=False)
    
    # Add coefficient values on the bars
    for i, row in enumerate(top_features.itertuples()):
        # Format coefficient value with 3 decimal places
        coef_text = f"{row.coefficient:.3f}"
        
        # Special case for "Products in use" - keep outside
        if row.feature == "Products in use":
            # Position text based on whether coefficient is positive or negative
            if row.coefficient < 0:
                # For negative values, place text to the left of the bar
                text_x = row.coefficient - 0.01
                ha = 'right'
                color = 'black'
            else:
                # For positive values, place text to the right of the bar
                text_x = row.coefficient + 0.01
                ha = 'left'
                color = 'black'
        else:
            # For all other features, place text inside the bar
            if row.coefficient < 0:
                # For negative values, place text inside bar toward right end
                text_x = row.coefficient / 2  # Middle of the bar
                ha = 'center'
                color = 'black'  # Dark text on light blue
            else:
                # For positive values, place text inside bar toward left end
                text_x = row.coefficient / 2  # Middle of the bar
                ha = 'center'
                color = 'black'  # Dark text on light orange
        
        # Add the text annotation
        ax.text(text_x, i, coef_text, va='center', ha=ha, fontsize=10, color=color)
    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.xlabel('Coefficient Value', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    
    # Move legend to a better position
    plt.legend(title="Effect on Churn", loc="lower right")
    
    plt.tight_layout()
    plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
    
    # 2. Combined ROC Curve and Confusion Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # ROC Curve
    ax1.plot(fpr, tpr, label=f'AUC = {tuned_auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontsize=14)
    ax1.set_title('ROC Curve', fontsize=16)
    ax1.legend(loc='lower right')
    ax1.grid(True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_roc)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
    disp.plot(ax=ax2, cmap='Blues', values_format='d')
    ax2.set_title('Confusion Matrix', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('logistic_regression_combined_plot.png', dpi=300, bbox_inches='tight')
    
    # Print practical suggestions based on the model
    print("\n" + "="*50)
    print("PRACTICAL INSIGHTS BASED ON MODEL RESULTS")
    print("="*50)
    
    # Get top positive and negative features
    top_pos = feature_importance[feature_importance['coefficient'] > 0].head(5)
    top_neg = feature_importance[feature_importance['coefficient'] < 0].head(5)
    
    print("\nFactors that INCREASE customer churn risk:")
    for idx, row in top_pos.iterrows():
        print(f"- {row['feature']}: coefficient = {row['coefficient']:.4f}")
    
    print("\nFactors that DECREASE customer churn risk:")
    for idx, row in top_neg.iterrows():
        print(f"- {row['feature']}: coefficient = {row['coefficient']:.4f}")
    
    return best_model, feature_importance, (basic_auc, basic_recall, tuned_auc, tuned_recall)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model and get results
    model, feature_importance, metrics = train_model(df)
    
    return model

if __name__ == "__main__":
    model = main() 