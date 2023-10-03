import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Introduction to KNN

def generate_data(n_samples=200):
    """Generate a 2D binary classification dataset."""
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1)
    return X, y

def plot_data(X, y):
    """Visualize the dataset."""
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Sample Dataset for KNN')
    plt.show()

if __name__ == "__main__":
    X, y = generate_data()
    plot_data(X, y)
