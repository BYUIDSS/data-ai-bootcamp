import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Generate the spiral dataset (similar to previous lessons)
def generate_spiral_data(n_points, noise=1.0):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360

    # Spiral 1
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise

    # Spiral 2 - Offset by 2 * pi/3
    d2x = -np.cos(n + 2 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise
    d2y = np.sin(n + 2 * np.pi / 3) * n + np.random.rand(n_points, 1) * noise

    # Spiral 3 - Offset by 4 * pi/3
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

    # Train a Decision Tree model using sklearn
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    # Predictions
    y_pred = dt.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.2f}")

    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    # Train a Random Forest model using sklearn
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_rf = clf.predict(X_test)

    # Calculate accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {accuracy_rf:.2f}")

    # Confusion matrix and classification report for Random Forest
    conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
    class_report_rf = classification_report(y_test, y_pred_rf)
    print("Confusion Matrix for Random Forest:")
    print(conf_matrix_rf)
    print("Classification Report for Random Forest:")
    print(class_report_rf)

    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    print("Best Hyperparameters for Random Forest:")
    print(best_params)
    
    # Predictions with the tuned Random Forest
    y_pred_tuned = grid_search.predict(X_test)

    # Calculate accuracy for the tuned model
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    print(f"Tuned Random Forest Accuracy: {accuracy_tuned:.2f}")
