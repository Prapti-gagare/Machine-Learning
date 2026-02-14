import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Iris.csv")
X = data.iloc[:, 0:4].values

n, m = X.shape

def calc_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def kmeans_wcss(X, k, max_iter=10):
    n = X.shape[0]

    random_index = np.random.choice(n, k, replace=False)
    centroids = X[random_index]

    for _ in range(max_iter):
        labels = []

        for i in range(n):
            distances = [calc_distance(X[i], c) for c in centroids]
            labels.append(np.argmin(distances))

        labels = np.array(labels)

        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

    wcss = 0
    for i in range(n):
        centroid = centroids[labels[i]]
        wcss += np.sum((X[i] - centroid) ** 2)

    return wcss


k_values = range(1, 11)
wcss_values = []

for k in k_values:
    wcss = kmeans_wcss(X, k)
    wcss_values.append(wcss)

plt.plot(k_values, wcss_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
