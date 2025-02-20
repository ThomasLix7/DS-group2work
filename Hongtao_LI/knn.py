import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read the CSV file
df = pd.read_csv('Branch1.csv')

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

# Scale the features (very important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
# Using square root of n as a rule of thumb for choosing k
k = int(np.sqrt(len(X_train)))
if k % 2 == 0:  # Make sure k is odd to avoid ties
    k += 1

model = KNeighborsClassifier(
    n_neighbors=k,
    weights='uniform',
    metric='euclidean'
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print(f"Number of neighbors (k): {k}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Find optimal k (optional)
print("\nFinding optimal k...")
k_range = range(1, 31, 2)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    score = knn.score(X_test_scaled, y_test)
    k_scores.append(score)

# Print best k
best_k = k_range[np.argmax(k_scores)]
print(f"\nBest k: {best_k}")
print(f"Best accuracy: {max(k_scores):.4f}")
