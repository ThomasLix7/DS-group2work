import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


# Load and preprocess data
df = pd.read_csv("QM_pre-process/output.csv")
df = df.drop(['Customer_ID', 'Source'], axis=1)

# Prepare features (X) and target (y)
X = df.drop('Left', axis=1)
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# Scale the features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GridSearchCV 
knn_model = KNeighborsClassifier()

param_grid = {
    'n_neighbors': range(1, 31, 2),  # Test odd numbers 1-29
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # For Minkowski distance
}

grid_search = GridSearchCV(
    knn_model, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1, 
    verbose=1
)

# Fit on scaled data
grid_search.fit(X_train_scaled, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-val accuracy: {grid_search.best_score_:.4f}")

# Evaluate on test set using scaled data
y_pred = best_model.predict(X_test_scaled)

# Print results
print("\nTest Set Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the best parameters
best_params = grid_search.best_params_
print("\nBest Parameters for future use:")
print(best_params)