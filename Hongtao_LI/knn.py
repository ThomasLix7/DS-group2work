import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

df1 = pd.read_csv('Branch1.csv')
df2 = pd.read_csv('Branch2.csv')
df3 = pd.read_csv('Branch3.csv')

# Combine all branches
df = pd.concat([df1, df2, df3], ignore_index=True).copy()

# Encode categorical variables FIRST
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Convert to numerical before imputation

# Now define features (Gender is numerical)
features = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']

# Then check missing values
missing_cols = df[features].columns[df[features].isna().any()].tolist()
print(f"Columns with missing values: {missing_cols}")

# Fix 3: Use proper imputation (safer method)
df = df.fillna(df[features].mean())

# Prepare features (X) and target (y)
X = df[features]
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Remove manual k selection and simple pipeline
# Replace with GridSearchCV pipeline
full_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    KNeighborsClassifier()
)

param_grid = {
    'kneighborsclassifier__n_neighbors': range(1, 31, 2),  # Test odd numbers 1-29
    'kneighborsclassifier__weights': ['uniform', 'distance'],
    'kneighborsclassifier__metric': ['euclidean', 'manhattan', 'minkowski'],
    'kneighborsclassifier__p': [1, 2]  # For Minkowski distance
}

grid_search = GridSearchCV(full_pipeline, param_grid, 
                          cv=5, scoring='accuracy',
                          n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))