import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools

# Load data
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Hyperparameter ranges
n_estimators_list = [50, 100, 150]
max_depth_list = [None, 3, 5, 7]
min_samples_split_list = [2, 4]
min_samples_leaf_list = [1, 2]
max_features_list = ['sqrt', 'log2', None]

best_accuracy = 0
best_params = {}
accuracy_records = []  # Store all combinations with their accuracy

# Manual grid search using nested loops
for n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features in itertools.product(
    n_estimators_list, max_depth_list, min_samples_split_list, min_samples_leaf_list, max_features_list
):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Store combination and accuracy
    accuracy_records.append({
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'accuracy': accuracy
    })

    # Update best
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }

# Print best parameters
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)

# Print all accuracies in descending order
accuracy_records_sorted = sorted(accuracy_records, key=lambda x: x['accuracy'], reverse=True)
print("\nAll hyperparameter combinations sorted by accuracy:")
for record in accuracy_records_sorted:
    print(record)
