def f(x, y):
    return x*x + y*y
def dfx(x, y):
    return 2*x

def dfy(x, y):
    return 2*y

def within_tolerance(x_old, y_old, x_new, y_new, eps):
    return abs(x_new - x_old) < eps and abs(y_new - y_old) < eps

def gradient_descent(x_old, y_old, lr, max_itr, eps):
    itr = 0
    close_enough = False

    while itr < max_itr and not close_enough:
        grad_x = dfx(x_old, y_old)
        grad_y = dfy(x_old, y_old)

        x_new = x_old - lr * grad_x
        y_new = y_old - lr * grad_y

        print(f"Iteration {itr}: x = {x_new}, y = {y_new}, f(x,y) = {f(x_new, y_new)}")

        close_enough = within_tolerance(x_old, y_old, x_new, y_new, eps)

        x_old = x_new
        y_old = y_new
        itr += 1

    print("Minimum is approximately at:")
    print("x =", x_new)
    print("y =", y_new)
gradient_descent(5, 4, 0.1, 100, 1e-6)
