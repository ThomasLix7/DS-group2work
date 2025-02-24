import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

# Load and clean data
df = pd.read_csv("Branch1.csv")

# Convert 'Gender' to categorical
df['Gender'] = df['Gender'].astype('category')
df = df.copy()  # Create a copy to avoid the chained assignment warning
df = df.dropna()  # Drop any remaining NaN values

# Feature engineering
df['Score/Age'] = df['Score'] / df['Age']
df['Products/Balance'] = df['Products_in_Use'] / (df['Balance'] + 1)
df['Score*Tenure'] = df['Score'] * df['Tenure']  # Interaction term
df['Balance/Age'] = df['Balance'] / (df['Age'] + 1)  # New ratio feature

# Define features
numeric_features = ['Score', 'Age', 'Tenure', 'Salary', 'Balance', 
                   'Products_in_Use', 'Score/Age', 'Products/Balance', 'Score*Tenure', 'Balance/Age']
categorical_features = ['Gender']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('scaler', RobustScaler()),
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ]), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Split raw data first
X = df.drop(['Customer_ID', 'Left'], axis=1)
y = df['Left'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert y to pandas Series
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

# Create pipeline
logit_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(
        class_weight={0: 1, 1: 2},  # Manual class weighting
        C=0.5,  # Tighter regularization
        solver='saga',  # Allows elasticnet
        penalty='elasticnet',
        l1_ratio=0.5,
        max_iter=2000
    ))
])

# Update parameter grid to only use compatible solvers
param_grid = {
    'model__C': [0.01, 0.1, 1],
    'model__solver': ['saga']  # Only saga supports elasticnet
}

# Perform GridSearchCV
logit_gs = GridSearchCV(logit_pipe, param_grid, scoring='roc_auc', cv=5)

try:
    # Fit the model
    logit_gs.fit(X_train, y_train)
    
    print("\nBest parameters:", logit_gs.best_params_)
    print("Best cross-validation ROC AUC score:", logit_gs.best_score_)
    
    # Get predictions
    y_pred = logit_gs.predict(X_test)
    y_pred_proba = logit_gs.predict_proba(X_test)[:, 1]
    
    # Add threshold tuning
    probs = logit_pipe.fit(X_train, y_train).predict_proba(X_train)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Print metrics
    print("\nTest set results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance analysis
    feature_names = logit_gs.best_estimator_.named_steps['preprocessor'].get_feature_names_out()
    coefficients = logit_gs.best_estimator_.named_steps['model'].coef_[0]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)

    print("\nTop 10 Features:")
    print(importance_df.head(10))

    # Use custom threshold for predictions
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

except Exception as e:
    print("An error occurred during training:", str(e))
    raise 