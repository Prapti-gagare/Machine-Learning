import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Load Iris CSV
df = pd.read_csv("iris.csv")

# Features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert species to numbers
le = LabelEncoder()
y = le.fit_transform(y)

# Define CV methods
cv_methods = {
    "LOOCV": LeaveOneOut(),
    "K-Fold (5)": KFold(n_splits=5, shuffle=True, random_state=42),
    "Stratified K-Fold (5)": StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
}

# Hyperparameter values
k_values = range(1, 21)

final_results = []

for cv_name, cv in cv_methods.items():
    best_mean = 0
    best_std = 0
    best_k = 0
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        # Store best k for this CV method
        if mean_acc > best_mean:
            best_mean = mean_acc
            best_std = std_acc
            best_k = k
    
    final_results.append({
        "CV Method": cv_name,
        "Best k": best_k,
        "Mean Accuracy": best_mean,
        "Std Dev (Variance)": best_std
    })

# Convert to DataFrame
results_df = pd.DataFrame(final_results)

print("\nFinal Comparison of Cross-Validation Methods:\n")
print(results_df)