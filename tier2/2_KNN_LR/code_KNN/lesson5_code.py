import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Implementing KNN

def generate_data(n_samples=200):
    """Generate a 2D binary classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    return X, y

def knn_classifier(X_train, y_train, n_neighbors=3):
    """Train a KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

if __name__ == "__main__":
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = knn_classifier(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"KNN Classifier Accuracy: {accuracy * 100:.2f}%")
