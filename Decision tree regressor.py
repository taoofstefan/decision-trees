# Decision tree regressor
# Import libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel(); y[::5] += 3 * (0.5 - rng.rand(20))
# Defin regressor
regr = DecisionTreeRegressor(max_depth = 2)

# Train & test the model
regr.fit(X, y)
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]; y_1 = regr.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor ="black", c="yellow", label="data")
plt.plot(X_test, y_1, color ="red", label="Regressor", linewidth=2)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Decision Tree")
plt.legend()
plt.show()