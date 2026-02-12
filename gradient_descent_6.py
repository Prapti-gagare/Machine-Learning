import numpy as np
X = np.array([
    [0.402262, 0.352496],
    [0.867487, 0.0575684],
    [0.03909,  0.579944],
    [0.591981, 0.07738],
    [0.9257,   0.525091],
    [0.72213,  0.979145],
    [0.649659, 0.2901],
    [0.049676, 0.418382],
    [0.875639, 0.300000],   
    [0.389704, 0.450000],   
])

target = np.array([11.7694, 2.8541, 17.1778, 6.8166, 4.0622,
                   9.55, 7.6245, 15.8547, 3.0675, 12.0985])

X = np.hstack((np.ones((X.shape[0], 1)), X))   

def predict(X, weights):
    return np.dot(X, weights)

def objective_function(target, predicted):
    return np.mean((target - predicted) ** 2)

def update_weights(X, weights, target, predicted, learning_rate):
    n = len(target)
    gradients = (-2/n) * np.dot(X.T, (target - predicted))
    new_weights = weights - learning_rate * gradients
    return new_weights

def with_tolerance(old_error, new_error, epsilon):
    return abs(old_error - new_error) < epsilon

def linear_regression(X, target, learning_rate, max_iterations, epsilon):
    n, m = X.shape
    weights = np.random.uniform(-1, 1, size=m)

    old_error = 1e10
    iteration = 0

    while iteration < max_iterations:
        predicted = predict(X, weights)
        new_error = objective_function(target, predicted)

        if with_tolerance(old_error, new_error, epsilon):
            print("Converged at iteration:", iteration)
            break

        weights = update_weights(X, weights, target, predicted, learning_rate)

        print(f"Iteration {iteration}, Error = {new_error}")
        old_error = new_error
        iteration += 1

    return predicted, weights

predicted_output, converged_weights = linear_regression(
    X, target, learning_rate=0.01, max_iterations=100, epsilon=1e-6
)

print("\nConverged Weights:", converged_weights)
print("\nPredicted Output:", predicted_output)

predicted_first = np.dot(X[0], converged_weights)
print("\nPrediction for first X:", predicted_first)
