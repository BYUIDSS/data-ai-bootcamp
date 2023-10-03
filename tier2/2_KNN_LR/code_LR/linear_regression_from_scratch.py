import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Generate the spiral dataset
def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    
    # Spiral 1
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    
    # Spiral 2 - Offset by 2*pi/3
    d2x = -np.cos(n + 2*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    d2y = np.sin(n + 2*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    
    # Spiral 3 - Offset by 4*pi/3
    d3x = -np.cos(n + 4*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    d3y = np.sin(n + 4*np.pi/3) * n + np.random.rand(n_points, 1) * noise
    
    # Combine the spirals and assign labels
    data = np.vstack((np.hstack((d1x, d1y, np.ones((n_points, 1)))), 
                      np.hstack((d2x, d2y, 2*np.ones((n_points, 1)))), 
                      np.hstack((d3x, d3y, 3*np.ones((n_points, 1))))))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['X', 'y', 'spiral'])
    
    return df

# Splitting the dataset into training and testing sets
def train_test_split(X, y, test_size=0.2):
    data = np.hstack((X, y.reshape(-1, 1)))
    np.random.shuffle(data)
    test_rows = int(test_size * data.shape[0])
    X_train, X_test = data[:-test_rows, :-1], data[-test_rows:, :-1]
    y_train, y_test = data[:-test_rows, -1], data[-test_rows:, -1]
    return X_train, X_test, y_train, y_test

# Linear Regression from Scratch
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights, self.bias = None, None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Generate the dataset
    df = generate_spiral_data(300)
    X = df[['X', 'y']].values
    y = df['spiral'].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the Linear Regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predictions
    y_pred = regressor.predict(X_test)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Visualization of Training Data and Regression Line
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], y_train, color='blue', label='Training Data')
    plt.plot(X_train[:, 0], regressor.predict(X_train), color='red', label='Regression Line')
    plt.title('Training Data and Regression Line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

    # Visualization of Testing Data and Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], y_test, color='blue', label='True values')
    plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted values')
    plt.title('Testing Data and Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Visualization of Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(X_test[:, 0], residuals, color='green', label='Residuals')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residuals')
    plt.xlabel('X')
    plt.ylabel('Residual')
    plt.legend()
    plt.show()