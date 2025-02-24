import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer

def load_and_preprocess_data():
    # Load data from all branches
    df1 = pd.read_csv('Branch1.csv')
    df2 = pd.read_csv('Branch2.csv')
    df3 = pd.read_csv('Branch3.csv')
    
    # Combine all branches
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Drop Customer_ID as it's not relevant for prediction
    df = df.drop('Customer_ID', axis=1)
    
    # Convert Gender to numerical
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    numeric_columns = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use']
    df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
    
    print("\nFeature Statistics:")
    print(df.describe())
    
    print("\nClass Distribution:")
    print(df['Left'].value_counts(normalize=True))
    
    return df

def train_model(df):
    # Separate features and target
    X = df.drop('Left', axis=1)
    y = df['Left']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define different solver-penalty combinations to try
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
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(best_model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return best_model, metrics, feature_importance

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Train model and get results
    model, metrics, feature_importance = train_model(df)
    
    # Print results
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

if __name__ == "__main__":
    model = main() 