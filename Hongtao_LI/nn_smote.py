import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

# Load data from all branches
df1 = pd.read_csv('Branch1.csv')
df2 = pd.read_csv('Branch2.csv')
df3 = pd.read_csv('Branch3.csv')

# Combine all branches
df = pd.concat([df1, df2, df3], ignore_index=True)

# Handle missing values (new way)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
df['Score'] = df['Score'].fillna(df['Score'].median())
df['Balance'] = df['Balance'].fillna(df['Balance'].median())
df['Products_in_Use'] = df['Products_in_Use'].fillna(df['Products_in_Use'].median())
df['Age'] = df['Age'].fillna(df['Age'].median())

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Prepare features (X) and target (y)
features = ['Age', 'Score', 'Tenure', 'Salary', 'Balance', 'Products_in_Use', 'Gender']
X = df[features]
y = df['Left']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain class distribution in splits
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modify param_grid to include stronger regularization
param_grid = {
    'hidden_layer_sizes': [(32,16), (64,32), (64,32,16)],  # Try smaller networks
    'activation': ['relu', 'tanh'],
    'alpha': [0.001, 0.01, 0.1],  # Increase regularization strength
    'learning_rate_init': [0.001, 0.01],
    'batch_size': [128, 256]
}

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

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train_scaled, y_train)

# Create and run grid search with stratified cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=3),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X_res, y_res)

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
    best_nn, X_res, y_res,
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
    random_state=42
)

# Plot feature importance
importance_df = pd.DataFrame({
    'Feature': features,
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