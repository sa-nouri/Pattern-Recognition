import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Union
import os

def create_directory(path: str) -> None:
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, title: str = "Decision Boundary") -> None:
    """Plot decision boundary for 2D data."""
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize data using z-score normalization."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

def save_results(results: dict, filename: str) -> None:
    """Save results to a file."""
    create_directory('results')
    np.save(f'results/{filename}.npy', results)

def load_results(filename: str) -> dict:
    """Load results from a file."""
    return np.load(f'results/{filename}.npy', allow_pickle=True).item() 
