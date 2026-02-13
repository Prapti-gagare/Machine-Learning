import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
data = load_iris()
X = data.data
y = data.target
k = 5
folds = np.random.choice(k, size=X.shape[0], replace=True)
X_test = X[folds == 0]
y_test = y[folds == 0]
X_train = X[folds != 0]
y_train = y[folds != 0]
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3,
    splitter='best',
    min_samples_split=10,
    min_samples_leaf=2,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
accuracy_dict = {}
for depth in range(1, 11):
    for split in range(2, 10):
        for leaf in range(1, 5):
            clf = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_split=split,
                min_samples_leaf=leaf,
                random_state=42
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_dict[(depth, split, leaf)] = accuracy
best_params = max(accuracy_dict, key=accuracy_dict.get)
best_accuracy = accuracy_dict[best_params]

print("\nBest Parameters (depth, split, leaf):", best_params)
print("Best Accuracy:", best_accuracy)
