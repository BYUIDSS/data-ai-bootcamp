from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # Generate sample data
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Linear Regression without regularization
    regressor = LinearRegression()
    regressor.fit(X_train_scaled, y_train)
    print(f"Linear Regression Score: {regressor.score(X_test_scaled, y_test)}")
