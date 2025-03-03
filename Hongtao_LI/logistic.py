import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, precision_recall_curve, classification_report

def load_and_preprocess_data():
    # Load data
    df = pd.read_csv("QM_pre-process/output.csv")
    df = df.drop(['Customer_ID', 'Source'], axis=1)
    
    return df

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
                'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}]
            }
        else:
            param_grid = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:3}, {0:1, 1:4}]
            }
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=log_model,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
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

    # Find optimal threshold using ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Find optimal threshold using precision-recall curve
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    optimal_idx_pr = np.argmax(f1_scores[:-1])  # Last element of f1_scores may be undefined
    optimal_threshold_pr = thresholds_pr[optimal_idx_pr]

    print("\nThreshold Optimization:")
    print(f"ROC optimal threshold: {optimal_threshold:.3f}")
    print(f"PR optimal threshold: {optimal_threshold_pr:.3f}")

    # Make predictions with both thresholds
    y_pred_roc = (y_pred_proba >= optimal_threshold).astype(int)
    y_pred_pr = (y_pred_proba >= optimal_threshold_pr).astype(int)

    print("\nMetrics with ROC optimal threshold:")
    print(classification_report(y_test, y_pred_roc))

    print("\nMetrics with PR optimal threshold:")
    print(classification_report(y_test, y_pred_pr))
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return best_model, feature_importance

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model and get results
    model, feature_importance = train_model(df)
    
    # Print results
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    model = main() 