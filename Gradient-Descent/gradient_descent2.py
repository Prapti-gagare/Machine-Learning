
x = 10         
lr = 0.1       
iterations = 20
for i in range(iterations):
    grad = 2 * x          
    x = x - lr * grad     
    print(f"Iteration {i+1}: x = {x}")

print("Minimum value of x is approximately:", x)
