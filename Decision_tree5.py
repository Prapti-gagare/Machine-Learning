import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    feature_names=data.feature_names,   
    class_names=data.target_names,      
    filled=True,
    rounded=True,                       
    fontsize=10
)
plt.show()
