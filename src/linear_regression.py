import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.w = 0.0
        self.b = 0.0

        for _ in range(self.epochs):
            y_pred = self.w * X + self.b

            dw = (1 / n_samples) * np.sum((y_pred - y) * X)
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return self.w * X + self.b
