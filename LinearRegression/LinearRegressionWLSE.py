import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\Elif\\Downloads\\salary dataset - simple linear regression.zip")


X = data["YearsExperience"].values.reshape(-1, 1)
y = data["Salary"].values.reshape(-1, 1)

X_b = np.hstack([np.ones((X.shape[0], 1)), X])


theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

y_pred = X_b.dot(theta_best)

mse = np.mean((y - y_pred) ** 2)

plt.scatter(X, y, color="blue", label="Gerçek Değerler")
plt.plot(X, y_pred, color="red", label="Tahmin Edilen Doğru")
plt.xlabel("İş Deneyimi (Yıl)")
plt.ylabel("Maaş")
plt.title("Manuel Linear Regression (Least Squares)")
plt.legend()
plt.show()

print(f"θ0 (Bias): {theta_best[0][0]:.2f}")
print(f"θ1 (Katsayı): {theta_best[1][0]:.2f}")
print(f"Mean Squared Error: {mse:.2f}")


