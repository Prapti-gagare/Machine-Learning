def f(x):
    f_x = ((1.4 - (x + 0.64 * 0.5))**2 +
           (1.9 - (x + 0.64 * 2.3))**2 +
           (3.2 - (x + 0.64 * 2.9))**2)
    return f_x

def df(x):
    grad = (2 * (1.4 - (x + 0.64 * 0.5)) * (-1) +
            2 * (1.9 - (x + 0.64 * 2.3)) * (-1) +
            2 * (3.2 - (x + 0.64 * 2.9)) * (-1))
    return grad

def within_tolerance(x_old, x_new, eps):
    return abs(x_new - x_old) < eps

def grad_descent(x_old, learning_rate, max_itr, eps):
    itr_no = 0
    close_enough = False

    while itr_no < max_itr and not close_enough:
        gradient = df(x_old)
        x_new = x_old - learning_rate * gradient
        
        print("Iteration:", itr_no,
              "x =", x_new,
              "f(x) =", f(x_new))

        close_enough = within_tolerance(x_old, x_new, eps)
        x_old = x_new
        itr_no += 1

    print("\nMinimum of the given function is at x =", x_new)
    print("Minimum value =", f(x_new))
grad_descent(0, 0.1, 100, 1e-7)
