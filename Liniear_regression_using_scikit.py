from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
Y = np.array([2,4,5,4,5])

model = LinearRegression()
model.fit(X, Y)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

prediction = model.predict([[6]])

print("Prediction:", prediction)