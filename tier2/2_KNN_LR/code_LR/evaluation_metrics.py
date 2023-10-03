import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

if __name__ == "__main__":
    # Sample data for testing
    y_true = [3, 2, 4, 5]
    y_pred = [2.8, 2.3, 3.9, 5.2]

    print(f"Mean Absolute Error (MAE): {calculate_mae(y_true, y_pred)}")
    print(f"Mean Squared Error (MSE): {calculate_mse(y_true, y_pred)}")
    print(f"Root Mean Squared Error (RMSE): {calculate_rmse(y_true, y_pred)}")
