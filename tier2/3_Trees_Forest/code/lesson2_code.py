import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Generate the spiral dataset (similar to lesson 1)
def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360

    # Spiral 1
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise

    # Spiral 2 - Offset by 2*pi/3
    d2x = -np.cos(n + 2 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise
    d2y = np.sin(n + 2 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise

    # Spiral 3 - Offset by 4*pi/3
    d3x = -np.cos(n + 4 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise
    d3y = np.sin(n + 4 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise

    # Combine the spirals and assign labels
    data = np.vstack((np.hstack((d1x, d1y, np.ones((n_points, 1))),
                          np.hstack((d2x, d2y, 2 * np.ones((n_points, 1))),
                          np.hstack((d3x, d3y, 3 * np.ones((n_points, 1)))))))

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['X', 'y', 'spiral'])

    return df

if __name__ == "__main__":
    # Generate the dataset
    df = generate_spiral_data(300)
    X = df[['X', 'y']].values
    y = df['spiral'].values

    # Split data into training and testing sets using sklearn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model using sklearn
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Visualization of Testing Data and Predictions
    plt.scatter(X_test[:, 0], y_test, color='blue', label='True values')
    plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted values')
    plt.title('Testing Data and Predictions using Random Forests')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
