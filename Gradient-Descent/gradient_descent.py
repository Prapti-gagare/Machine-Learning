# Gradient Descent for two variables: f(x,y) = x^2 + y^2

x = 5.0      # initial value of x
y = 4.0      # initial value of y
lr = 0.1     # learning rate
eps = 0.001  # stopping criteria

while True:
    x_old = x
    y_old = y

    # derivatives
    dx = 2 * x
    dy = 2 * y

    # x_new and y_new formula
    x = x - lr * dx
    y = y - lr * dy

    print("x =", x, " y =", y)

    # stopping condition
    if abs(x - x_old) < eps and abs(y - y_old) < eps:
        break

print("\nMinimum point is approximately:")
print("x =", x, " y =", y)
