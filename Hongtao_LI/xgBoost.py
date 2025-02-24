import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file
df1 = pd.read_csv('Branch1.csv')
df2 = pd.read_csv('Branch2.csv')
df3 = pd.read_csv('Branch3.csv')

# Print data distribution before combining
print("\nData distribution in branches:")
print(f"Branch1 Left=1: {df1['Left'].mean():.2%}")
print(f"Branch2 Left=1: {df2['Left'].mean():.2%}")
print(f"Branch3 Left=1: {df3['Left'].mean():.2%}")

    # Combine all branches
df = pd.concat([df1, df2, df3], ignore_index=True)

# Handle missing values
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Prepare features (X) and target (y)
features = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']
X = df[features]
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model with Grid Search
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

# Get best model
model = grid_search.best_estimator_

# Print grid search results
print("\nBest Parameters:", grid_search.best_params_)
print("Best CV Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))