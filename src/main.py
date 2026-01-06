import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100) * 10
y = 2 * X + 3 + np.random.randn(100)

# Train model
model = LinearRegression(learning_rate=0.01, epochs=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot results
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', label='Model')
plt.legend()
plt.show()
