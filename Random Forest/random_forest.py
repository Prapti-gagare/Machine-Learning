import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
x = data.data
y = data.target

k = 3
folds = np.random.choice(k, size=x.shape[0], replace=True)

x_test = x[folds == 0]
y_test = y[folds == 0]

x_train = x[folds != 0]
y_train = y[folds != 0]

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=100,     
    random_state=42
)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
y_prob = clf.predict_proba(x_test)

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
mcc = matthews_corrcoef(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)
