from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, KFold, StratifiedKFold, cross_val_score
import numpy as np
import time

datasets = {
    "Iris": load_iris(return_X_y=True),
    "Breast Cancer": load_breast_cancer(return_X_y=True)
}

k_values = [1, 3, 5, 7, 9]

def evaluate_cv(X, y, cv, name):
    results = []
    start_time = time.time()

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        results.append((k, mean_score, std_score))

        print(f"k = {k}")
        print(f"Mean Accuracy = {mean_score:.4f}")
        print(f"Std Dev (Variance) = {std_score:.4f}")
        print("-"*30)

    total_time = time.time() - start_time
    best_k = max(results, key=lambda x: x[1])[0]

    print(f"Best k ({name}) = {best_k}")
    print(f"Time taken ({name}) = {total_time:.4f} seconds")

for dataset_name, (X, y) in datasets.items():
    print("\n" + "="*60)
    print(f"DATASET: {dataset_name}")
    print("="*60)

    print("\nLOOCV Results")
    loo = LeaveOneOut()
    evaluate_cv(X, y, loo, "LOOCV")

    print("\n10-Fold CV Results")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    evaluate_cv(X, y, kf, "10-Fold CV")

    print("\nStratified 10-Fold CV Results")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    evaluate_cv(X, y, skf, "Stratified 10-Fold CV")