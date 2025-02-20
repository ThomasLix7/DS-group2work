import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve

# Read the CSV file
df = pd.read_csv('Branch1.csv')

# Handle missing values (new way)
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

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

# Train neural network
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=17,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    learning_rate_init=0.001
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance through permutation
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model,  # Pass the model directly instead of lambda
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

# Add cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Plot learning curves
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5
)