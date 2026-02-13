import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
accuracies = []
precisions = []
recalls = []
f1_scores = []
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,splitter='best',min_samples_split=10,min_samples_leaf=2,random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average='macro'))
    recalls.append(recall_score(y_test, y_pred, average='macro'))
    f1_scores.append(f1_score(y_test, y_pred, average='macro'))
    print(f"\nFold {fold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n===== 5-Fold Cross Validation Results =====")
print("Average Accuracy :", np.mean(accuracies))
print("Average Precision:", np.mean(precisions))
print("Average Recall   :", np.mean(recalls))
print("Average F1 Score :", np.mean(f1_scores))
