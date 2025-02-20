import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

# Load and clean data
df = pd.read_csv("Branch1.csv")

# Convert 'Gender' to categorical
df['Gender'] = df['Gender'].astype('category')
df = df.copy()  # Create a copy to avoid the chained assignment warning
df = df.dropna()  # Drop any remaining NaN values

# Feature engineering
df['Score/Age'] = df['Score'] / df['Age']
df['Products/Balance'] = df['Products_in_Use'] / (df['Balance'] + 1)

# Define features
numeric_features = ['Score', 'Age', 'Tenure', 'Salary', 'Balance', 
                   'Products_in_Use', 'Score/Age', 'Products/Balance']
categorical_features = ['Gender']

# Create preprocessor
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numeric_features),
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
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Parameter grid
param_grid = {
    'model__C': [0.01, 0.1, 1],
    'model__solver': ['lbfgs', 'saga']
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
    
    # Print metrics
    print("\nTest set results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance analysis
    feature_names = (numeric_features + 
                    [f"{f}_encoded" for f in categorical_features])
    coefficients = logit_gs.best_estimator_.named_steps['model'].coef_[0]
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    print("\nFeature Importance:")
    print(importance_df.sort_values('Coefficient', key=abs, ascending=False))

except Exception as e:
    print("An error occurred during training:", str(e))
    raise 