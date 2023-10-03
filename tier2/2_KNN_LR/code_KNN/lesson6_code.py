import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Evaluating and Improving KNN

def generate_data(n_samples=200):
    """Generate a 2D binary classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluate the KNN model."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return report

if __name__ == "__main__":
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    
    report = evaluate_model(model, X_test, y_test)
    print(report)
