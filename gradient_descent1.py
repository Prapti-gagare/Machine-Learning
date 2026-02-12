def gradient_descent(lr, x_init, iterations=50):
    x = x_init
    for i in range(iterations):
        grad = 2 * (x + 5)     
        x = x - lr * grad
    return x


learning_rates = [0.01, 0.05, 0.1, 0.5]
x0 = 10   

for lr in learning_rates:
    x_final = gradient_descent(lr, x0)
    print(f"Learning rate = {lr}, Converged x ≈ {x_final}")
