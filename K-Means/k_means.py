import numpy as np
import pandas as pd

data = pd.read_csv("Iris.csv")

X = data.iloc[:, 0:4].values   

k = 3
n, m = X.shape

def calc_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

random_index = np.random.choice(n, k, replace=False)
centroids = X[random_index]

print("Initial Centroids:")
print(centroids)
print()

max_iter = 10

for iteration in range(max_iter):
    distance0, distance1, distance2 = [], [], []
    for i in range(n):
        distance0.append(calc_distance(X[i], centroids[0]))
        distance1.append(calc_distance(X[i], centroids[1]))
        distance2.append(calc_distance(X[i], centroids[2]))

    labels = []
    for i in range(n):
        labels.append(np.argmin([distance0[i], distance1[i], distance2[i]]))

    labels = np.array(labels)

    cl0 = X[labels == 0]
    cl1 = X[labels == 1]
    cl2 = X[labels == 2]

    if len(cl0) > 0:
        centroids[0] = np.mean(cl0, axis=0)
    if len(cl1) > 0:
        centroids[1] = np.mean(cl1, axis=0)
    if len(cl2) > 0:
        centroids[2] = np.mean(cl2, axis=0)

    print(f"Iteration {iteration + 1} completed")

print("\nFinal Centroids:")
print(centroids)

print("\nCluster sizes:")
print("Cluster 0:", len(cl0))
print("Cluster 1:", len(cl1))
print("Cluster 2:", len(cl2))
