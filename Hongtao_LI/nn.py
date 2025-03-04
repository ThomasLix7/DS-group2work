import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

# Load and preprocess data
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
    stratify=y  # Maintain class distribution in splits
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modify param_grid to include stronger regularization
param_grid = {
    'hidden_layer_sizes': [(32,16), (64,32), (64,32,16), (128,64,32), (256,128,64)],  # Try different network structures
    'activation': ['relu', 'tanh'],
    'alpha': [0.001, 0.01, 0.1],  # Increase regularization strength
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [128, 256],
    'max_iter': [1000, 2000],
    'solver': ['adam', 'lbfgs']
}

# best model parameters
# nn_model = MLPClassifier(
#     hidden_layer_sizes=(128, 64, 32),
#     activation='tanh',               
#     solver='adam',
#     alpha=0.01,                        
#     learning_rate_init=0.001,
#     batch_size=128,
#     max_iter=1000,
#     early_stopping=True,
#     validation_fraction=0.2,
#     random_state=17
# )

# Initialize model
model = MLPClassifier(
    solver='adam',
    max_iter=1000,
    random_state=17,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    alpha=0.01
)

# Create and run grid search with stratified cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_train_scaled, y_train)

# Get best model
best_nn = grid_search.best_estimator_

# Print best parameters and score
print("\nBest parameters:", grid_search.best_params_)
print("Best CV ROC AUC:", grid_search.best_score_)

# Calculate cross-validation scores using the best model
cv = StratifiedKFold(n_splits=5)
cv_scores = cross_val_score(best_nn, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Calculate learning curve using the best model
train_sizes, train_scores, val_scores = learning_curve(
    best_nn, X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=StratifiedKFold(n_splits=5),
    scoring='roc_auc'
)

# Calculate feature importance
result = permutation_importance(
    best_nn,
    X_test_scaled, 
    y_test,
    n_repeats=10,
    random_state=17
)

# Plot feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
})
importance_df = importance_df.sort_values('Importance', ascending=True)

# Make predictions and evaluate
y_pred_proba = best_nn.predict_proba(X_test_scaled)
y_pred = best_nn.predict(X_test_scaled)

# Print results
print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Add balanced accuracy and ROC AUC metrics
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, best_nn.predict_proba(X_test_scaled)[:, 1]))

# Optional: Compare with training performance
y_train_pred = best_nn.predict(X_train_scaled)
print("\nTraining Set Accuracy:", accuracy_score(y_train, y_train_pred))