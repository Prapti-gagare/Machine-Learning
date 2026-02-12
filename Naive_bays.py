import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

data = pd.read_csv("data.csv")

x = data.iloc[:, :-1].values    
y = data.iloc[:, -1].values    
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in kf.split(x):
    folds = np.ones(len(x))
    folds[test_index] = 0
    break

X_train = x[folds != 0]
y_train = y[folds != 0]

X_test  = x[folds == 0]
y_test  = y[folds == 0]

prior_probs = pd.Series(y_train).value_counts(normalize=True)

def PDF(x, mean, var):
    var = var + 1e-9  
    first = 1 / np.sqrt(2 * np.pi * var)
    second = np.exp(-(x - mean) ** 2 / (2 * var))
    return first * second


mean_class0 = X_train[y_train == 0].mean(axis=0)
var_class0  = X_train[y_train == 0].var(axis=0)

mean_class1 = X_train[y_train == 1].mean(axis=0)
var_class1  = X_train[y_train == 1].var(axis=0)

y_pred = []
probs = []

for i in range(X_test.shape[0]):
    likelihood_class0 = PDF(X_test[i], mean_class0, var_class0)
    likelihood_class1 = PDF(X_test[i], mean_class1, var_class1)

    post0 = np.prod(likelihood_class0) * prior_probs[0]
    post1 = np.prod(likelihood_class1) * prior_probs[1]

    denom = post0 + post1
    post0 /= denom
    post1 /= denom

    probs.append([post0, post1])
    y_pred.append(np.argmax([post0, post1]))

cm = confusion_matrix(y_test, y_pred)

tp = cm[0, 0]
tn = cm[1, 1]
fp = cm[1, 0]
fn = cm[0, 1]

accuracy  = (tp + tn) / cm.sum()
precision = tp / (tp + fp)
recall    = tp / (tp + fn)

print("Confusion Matrix:\n", cm)
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
