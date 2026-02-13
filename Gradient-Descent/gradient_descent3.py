# Simple Gradient Descent for y = mx + c

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

m = 0        # slope
c = 0        # intercept
lr = 0.01    # learning rate
eps = 0.001  # stopping value

while True:
    m_old = m
    c_old = c

    dm = 0
    dc = 0
    n = len(x)

    for i in range(n):
        y_pred = m*x[i] + c
        dm += (y_pred - y[i]) * x[i]
        dc += (y_pred - y[i])

    dm = dm * (2/n)
    dc = dc * (2/n)

    # x_new = x_old - learning_rate * derivative
    m = m - lr * dm
    c = c - lr * dc

    print("m =", m, "c =", c)

    # stopping condition
    if abs(m - m_old) < eps and abs(c - c_old) < eps:
        break

print("\nFinal values:")
print("Slope m =", m)
print("Intercept c =", c)
