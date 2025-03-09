import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report, roc_auc_score, recall_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("QM_pre-process/output.csv")
    df = df.drop(['Customer_ID', 'Source'], axis=1)
    
    return df

# New function to train and evaluate a basic model with default parameters
def train_basic_model(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("Training Basic Logistic Regression Model (Default Parameters)")
    print("="*50)
    
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
    print(f"Balanced Accuracy: {balanced_acc_optimal:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Recall Score: {recall_optimal:.4f}")
    
    print("\nClassification Report (Basic Model, Optimal threshold):")
    print(classification_report(y_test, y_pred_optimal))
    
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
    
    print("\n" + "="*50)
    print("Hyperparameter Tuning Process")
    print("="*50)
    
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
        print(f"\nTrying solver={solver}, penalty={penalty}")
        
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
            
            print(f"Best score for this combination: {grid_search.best_score_:.4f}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            # Update best model if current one is better
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_solver_penalty = (solver, penalty)
        except Exception as e:
            print(f"Error with solver={solver}, penalty={penalty}: {str(e)}")
            continue
    
    print(f"\nBest overall solver-penalty combination: {best_solver_penalty}")
    print(f"Best overall parameters: {best_params}")
    print(f"Best overall score: {best_score:.4f}")
    
    # Make predictions with best model
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Find optimal threshold using ROC curve (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold_roc = thresholds[optimal_idx]
    
    # Find optimal threshold for maximizing recall (with minimum precision constraint)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    # Find threshold that gives at least 0.3 precision
    valid_indices = precisions[:-1] >= 0.3  # Exclude the last point
    if np.any(valid_indices):
        valid_recalls = recalls[:-1][valid_indices]
        valid_thresholds = thresholds_pr[valid_indices]
        # Get threshold with highest recall among valid thresholds
        best_recall_idx = np.argmax(valid_recalls)
        optimal_threshold_recall = valid_thresholds[best_recall_idx]
    else:
        # If no threshold meets the precision requirement, use ROC threshold
        optimal_threshold_recall = optimal_threshold_roc
    
    print("\nThreshold Optimization:")
    print(f"ROC optimal threshold (Youden's J): {optimal_threshold_roc:.3f}")
    print(f"Recall-optimized threshold: {optimal_threshold_recall:.3f}")
    
    # Calculate metrics with both thresholds
    y_pred_roc = (y_pred_proba >= optimal_threshold_roc).astype(int)
    y_pred_recall = (y_pred_proba >= optimal_threshold_recall).astype(int)
    
    # Calculate metrics
    tuned_auc = roc_auc_score(y_test, y_pred_proba)
    
    # ROC threshold metrics
    accuracy_roc = accuracy_score(y_test, y_pred_roc)
    balanced_acc_roc = balanced_accuracy_score(y_test, y_pred_roc)
    recall_roc = recall_score(y_test, y_pred_roc)
    
    # Recall-optimized threshold metrics
    accuracy_recall = accuracy_score(y_test, y_pred_recall)
    balanced_acc_recall = balanced_accuracy_score(y_test, y_pred_recall)
    recall_optimized = recall_score(y_test, y_pred_recall)
    
    print("\nMetrics with ROC optimal threshold:")
    print(f"Accuracy: {accuracy_roc:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_roc:.4f}")
    print(f"AUC: {tuned_auc:.4f}")
    print(f"Recall: {recall_roc:.4f}")
    print(classification_report(y_test, y_pred_roc))
    
    print("\nMetrics with Recall-optimized threshold:")
    print(f"Accuracy: {accuracy_recall:.4f}")
    print(f"Balanced Accuracy: {balanced_acc_recall:.4f}")
    print(f"AUC: {tuned_auc:.4f}")
    print(f"Recall: {recall_optimized:.4f}")
    print(classification_report(y_test, y_pred_recall))
    
    # Choose the threshold that gives the best balanced performance
    # We'll use the ROC threshold as it balances false positives and false negatives
    optimal_threshold = optimal_threshold_roc
    tuned_recall = recall_roc
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    # Add comparison of basic vs tuned model
    print("\n" + "="*50)
    print("BASIC VS FINE-TUNED MODEL PERFORMANCE COMPARISON")
    print("="*50)
    print(f"{'Metric':<15}{'Basic Model':<15}{'Fine-tuned Model':<15}{'Improvement':<15}")
    print(f"{'-'*60}")
    print(f"{'AUC':<15}{basic_auc:.4f}{'':<7}{tuned_auc:.4f}{'':<7}{((tuned_auc-basic_auc)/basic_auc)*100:.2f}%")
    print(f"{'Recall':<15}{basic_recall:.4f}{'':<7}{tuned_recall:.4f}{'':<7}{((tuned_recall-basic_recall)/basic_recall)*100:.2f}%")
    print(f"\nOptimal thresholds: Basic model: {basic_optimal_threshold:.3f}, Fine-tuned model: {optimal_threshold:.3f}")
    
    return best_model, feature_importance, (basic_auc, basic_recall, tuned_auc, tuned_recall)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model and get results
    model, feature_importance, metrics = train_model(df)
    
    # Print results
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    model = main() 