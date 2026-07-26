import numpy as np
X = np.array([1,2,3,4,5])
Y = np.array([2,4,5,4,5])

# Parameters
m = 0
c = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

for _ in range(epochs):

    # Predictions
    Y_pred = m * X + c

    # Gradients
    dm = (-2/n) * np.sum(X * (Y - Y_pred))
    dc = (-2/n) * np.sum(Y - Y_pred)

    # Update
    m = m - learning_rate * dm
    c = c - learning_rate * dc

print("Slope:", m)
print("Intercept:", c)