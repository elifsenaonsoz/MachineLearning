import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\Elif\\Downloads\\salary dataset - simple linear regression.zip")

X = data["YearsExperience"].values
y = data["Salary"].values


X_b = np.c_[np.ones((len(X), 1)), X.reshape(-1, 1)]
y = y.reshape(-1, 1)


learning_rate = 0.01
n_iterations = 1000
m = len(X)

theta = np.random.randn(2, 1)


cost_history = []

for iteration in range(n_iterations):
    gradients = (2/m) * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    cost = (1/m) * np.sum((X_b.dot(theta) - y) ** 2)
    cost_history.append(cost)

y_pred = X_b.dot(theta)


print(f"θ0 (Bias): {theta[0][0]:.2f}")
print(f"θ1 (Katsayı): {theta[1][0]:.2f}")
print(f"Final MSE: {cost_history[-1]:.2f}")

plt.plot(cost_history)
plt.title("Cost Fonksiyonunun Azalımı (Gradient Descent)")
plt.xlabel("İterasyon")
plt.ylabel("MSE")
plt.grid(True)
plt.show()

plt.scatter(X, y, color='blue', label='Gerçek Veriler')
plt.plot(X, y_pred, color='green', label='Gradient Descent Tahmini')
plt.xlabel("İş Deneyimi (Yıl)")
plt.ylabel("Maaş")
plt.title("Linear Regression - Gradient Descent")
plt.legend()
plt.show()
