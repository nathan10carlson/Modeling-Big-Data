import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# importing dataset
skiing = pd.read_excel('../data/skiing_dist.xlsx', index_col=0)
labels = skiing.index.to_numpy()
print(skiing.head())   # first 5 rows
print(skiing.shape)    # (rows, columns)
print(skiing.columns)  # column names

# distance matrix
Dist = skiing.to_numpy(dtype=float)

import numpy as np
import matplotlib.pyplot as plt


def classical_mds(Dist, d=2, labels=None, plot=True, figsize=(8, 6), point_color='blue'):
    """
    Classical MDS (Torgerson/Gower) from a distance matrix.

    Parameters
    ----------
    Dist : np.ndarray
        Square distance matrix of shape (n_samples, n_samples)
    d : int
        Target embedding dimension
    labels : list or np.ndarray
        Optional labels for plotting
    plot : bool
        Whether to generate a scatter plot
    figsize : tuple
        Figure size for plotting
    point_color : str
        Color for points in plot

    Returns
    -------
    Y : np.ndarray
        Embedded coordinates of shape (n_samples, d)
    eigvals : np.ndarray
        Eigenvalues of double-centered matrix
    """
    Dist = np.asarray(Dist)
    n = Dist.shape[0]
    if Dist.shape[0] != Dist.shape[1]:
        raise ValueError("Distance matrix must be square")

    # Step 1: double centering
    A = -0.5 * Dist ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = H @ A @ H

    # Step 2: eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)

    # Step 3: sort eigenvalues descending
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Step 4: take only positive eigenvalues
    tol = 1e-9
    pos = eigvals > tol
    eigvals = eigvals[pos]
    eigvecs = eigvecs[:, pos]

    if d > eigvals.size:
        print(f"Warning: requested {d} dimensions but only {eigvals.size} positive eigenvalues")
        d = eigvals.size

    # Step 5: compute configuration
    Lambda = np.diag(eigvals[:d])
    V = eigvecs[:, :d]
    configuration = (V @ np.sqrt(Lambda))  # shape (n_samples, d)

    # Optional plotting
    if plot:
        plt.figure(figsize=figsize)
        plt.scatter(configuration[:, 0], configuration[:, 1], color=point_color)
        if labels is not None:
            for i, label in enumerate(labels):
                plt.text(configuration[i, 0] + 0.02, configuration[i, 1] + 0.02, str(label), fontsize=9)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.title(f"{d}-D Classical MDS")
        plt.grid(True)
        plt.axis('equal')
        plt.show()

    return configuration, eigvals

classical_mds(Dist, d=2, labels=labels, plot=True, figsize=(8, 6), point_color='blue')