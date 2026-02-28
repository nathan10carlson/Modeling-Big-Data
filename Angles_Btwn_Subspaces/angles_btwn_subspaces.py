import numpy as np
from sympy import print_tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import random

from torch.fx.experimental.meta_tracer import manual_meta_overrides

#random.seed(0) # so i can reproduce
#np.random.seed(0)


def QR_decompose(matrix):
    Q, R = np.linalg.qr(matrix)
    return Q, R

def SVD_decompose(matrix):
    U, sing_vals, Vt = np.linalg.svd(matrix, full_matrices=False)
    #Sigma =np.diag(sing_vals)
    V = Vt.T
    return U, sing_vals, V

def angles_btwn_subspaces(X, Y, column_mean_subt=False):
    if column_mean_subt == True:
        X = X - np.mean(X, axis=0)
    Q_x, R_x = QR_decompose(X)
    Q_y, R_y = QR_decompose(Y)
    R_SIG_S_tp = Q_x.T @ Q_y
    R, sing_vals, S = SVD_decompose(R_SIG_S_tp)
    angles = np.arccos(sing_vals)  # returns radians
    return angles

class distance_calculation_func():
    def __init__(self):
        pass

    def geo_distance(self, X, Y):
        angles = angles_btwn_subspaces(X, Y)
        angles_squared = angles ** 2
        geo_dist = np.sum(angles_squared)
        return geo_dist

    def chordal_distance(self, X, Y):
        angles = angles_btwn_subspaces(X, Y)
        sin_sqd = np.sin(angles) ** 2
        chord_dist = np.sum(sin_sqd)
        return chord_dist

    def Fubini_Study_distance(self, X, Y):
        angles = angles_btwn_subspaces(X, Y)
        cos_angles = np.cos(angles)
        product = np.prod(cos_angles)
        arc_cos_prod = np.arccos(product)  # this is distance
        return arc_cos_prod

    def small_ang_pseudo_distance(self, X, Y):
        angles = angles_btwn_subspaces(X, Y)
        return np.min(angles)


# Practice from Kirby book
X = np.array([[2., 0.,1.,0.,1.],[1.,1.,0.,2.,1.],[10.,10.,770.,2.,1.]]).T
Y = np.array([[2.,1.,1.,1.,1.],[1.,1.,1.,0.,1.]]).T
#angles = angles_btwn_subspaces(X, Y)
#print(angles)




def toy_data(dimension, num_samples,
             manual_override=False,
             slope=None,
             y_intercept=None,
             noise_std= None):

    if manual_override:
        direction_vec = np.array([slope, y_intercept]).reshape(dimension, 1)
    else:
        direction_vec = np.random.randn(dimension, 1)

    scale = np.random.randn(num_samples, 1)

    # rank 1 sata
    data = scale @ direction_vec.T   # (num_samples, dimension)

    # Add Gaussian noise
    if noise_std > 0:
        noise = noise_std * np.random.randn(num_samples, dimension)
        data = data + noise

    return data, direction_vec

def generate_toy_data_classes(num_classes, dimension, num_samples,
             manual_override=False,
             slope=None,
             y_intercept=None,
             noise_std=None):
    data_plus_labels = []
    for i in range(num_classes):
        data, direction_vec = toy_data(dimension, num_samples,
             manual_override,
             slope,
             y_intercept, noise_std)
        label = i # class label
        to_append = [i, data, direction_vec]
        data_plus_labels.append(to_append)

    return data_plus_labels


def plot_toy_data_classes(data, title="Toy Data Classes"):
    """
    Plot 2D toy data with different colors for each class.

    Parameters:
    -----------
    data : list
        List of [label, data_array, direction_vec] for each class.
    title : str
        Plot title.
    """
    plt.figure(figsize=(6,6))

    # Automatically generate distinct colors
    num_classes = len(data)
    cmap = plt.get_cmap("tab10")  # up to 10 distinct colors
    colors = [cmap(i % 10) for i in range(num_classes)]

    for class_data in data:
        label = class_data[0]
        X_class = class_data[1]  # shape (num_samples, 2)
        direction = class_data[2]  # shape (2,1)

        # Plot the points
        plt.scatter(X_class[:,0], X_class[:,1],
                    color=colors[label], alpha=0.7, label=f"Class {label}")

        # Plot the generating direction vector
        plt.quiver(0, 0, direction[0,0], direction[1,0],
                   angles='xy', scale_units='xy', scale=1,
                   color=colors[label], width=0.01)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

# Gen toy data
data_w_labels = generate_toy_data_classes(num_classes=3, dimension=2, num_samples=4, noise_std=0.2)
print(data_w_labels)
# Plot
plot_toy_data_classes(data_w_labels)

import numpy as np

def stack_data_and_labels(data_w_labels):
    """
    Stack toy class data into a single array with labels.

    Parameters:
    -----------
    data_w_labels : list
        List of [label, data_array, direction_vec] for each class.

    Returns:
    --------
    X_all : np.ndarray
        Stacked data of shape (total_samples, dimension)
    y_all : np.ndarray
        Labels corresponding to each row in X_all
    """
    X_list = []
    y_list = []

    for class_data in data_w_labels:
        label = class_data[0]
        X_class = class_data[1]  # shape (num_samples, dimension)
        num_samples = X_class.shape[0]

        X_list.append(X_class)
        y_list.append(np.full(num_samples, label))

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    return X_all, y_all

X_all, y_all = stack_data_and_labels(data_w_labels)

print("X_all shape:", X_all.shape)  # (num_classes*num_samples, dimension)
print("y_all shape:", y_all.shape)  # (num_classes*num_samples,)
print("Labels:", np.unique(y_all))


def pca_from_scratch(X, n_components=None):
    """
    Perform PCA from scratch using eigen decomposition.

    Parameters:
    -----------
    X : np.ndarray, shape (n_samples, n_features)
        Data matrix.
    n_components : int or None
        Number of principal components to keep. If None, keep all.

    Returns:
    --------
    X_pca : np.ndarray, shape (n_samples, n_components)
        Projected data onto principal components.
    eigenvalues : np.ndarray
        Eigenvalues of the covariance matrix (sorted descending).
    eigenvectors : np.ndarray
        Corresponding eigenvectors (columns) sorted by eigenvalue.
    explained_var_ratio : np.ndarray
        Variance explained by each PC.
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)

    # Covariance matrix
    cov = np.cov(X_centered, rowvar=False)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # eigh for symmetric matrices

    # Sort eigenvalues and vectors descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project data
    if n_components is None:
        n_components = X.shape[1]
    X_pca = X_centered @ eigenvectors[:, :n_components]

    # Explained variance ratio
    explained_var_ratio = eigenvalues / np.sum(eigenvalues)

    return X_pca, eigenvalues, eigenvectors, explained_var_ratio

def plot_pca_and_energy(X, y, title="PCA from Scratch"):
    """
    Plot PCA first two components and explained variance (energy).
    """
    X_pca, eigenvalues, eigenvectors, explained_var_ratio = pca_from_scratch(X)

    # Scatter plot of first 2 PCs
    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)

    classes = np.unique(y)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(classes))]

    for i, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7, color=colors[i], label=f"Class {cls}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axis("equal")

    # Energy / explained variance plot
    plt.subplot(1, 2, 2)
    plt.bar(range(1, len(explained_var_ratio)+1), explained_var_ratio, alpha=0.7)
    plt.plot(range(1, len(explained_var_ratio)+1), np.cumsum(explained_var_ratio),
             marker='o', color='red', label="Cumulative")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Energy Plot")
    plt.xticks(range(1, len(explained_var_ratio)+1))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

#
#plot_pca_and_energy(X_all, y_all)
'''
def build_subspaces(data_pts, sbspc_size, without_replace=False):
    """
    Build subspaces by selecting sbspc_size columns,
    always including the i-th column.

    Parameters
    ----------
    data_pts : np.ndarray (n_features, n_samples)
    sbspc_size : int
    without_replace : bool
        Whether to sample additional columns without replacement

    Returns
    -------
    subspaces : list of np.ndarray
        Each element is a matrix (n_features, sbspc_size)
    """

    n_features, n_samples = data_pts.shape
    subspaces = []

    for i in range(n_samples):

        # Start subspace with column i
        subspace = np.zeros((n_features, sbspc_size))
        subspace[:, 0] = data_pts[:, i]

        # All possible indices except i
        candidate_indices = np.delete(np.arange(n_samples), i)

        # Randomly choose remaining columns
        chosen = np.random.choice(
            candidate_indices,
            size=sbspc_size - 1,
            replace=not without_replace
        )

        # Fill remaining columns
        for j, idx in enumerate(chosen):
            subspace[:, j + 1] = data_pts[:, idx]

        subspaces.append(subspace)

    return subspaces'''


def build_subspaces_fast(data_pts, sbspc_size, without_replace=False):
    """
    Faster version of subspace builder.

    data_pts: (n_features, n_samples)
    returns list of subspace matrices (n_features, sbspc_size)
    """

    n_features, n_samples = data_pts.shape

    if without_replace and sbspc_size > n_samples:
        raise ValueError("Subspace size too large for sampling without replacement.")

    subspaces = []
    all_indices = np.arange(n_samples)

    for i in range(n_samples):

        if without_replace:
            # remove i from candidates
            candidates = np.delete(all_indices, i)
            chosen = np.random.choice(
                candidates,
                size=sbspc_size - 1,
                replace=False
            )
        else:
            chosen = np.random.choice(
                all_indices,
                size=sbspc_size - 1,
                replace=True
            )

        # Combine i with chosen indices
        cols = np.concatenate(([i], chosen))

        # Direct column slicing (no inner loop!)
        subspace = data_pts[:, cols]

        subspaces.append(subspace)

    return subspaces

import numpy as np

def construct_dist_matrix(subspaces: np.ndarray, dist_type: str) -> np.ndarray:
    ## inputs as num_sub/samp, dim, sbs_sze
    if not isinstance(subspaces, np.ndarray):
        raise TypeError("subspaces must be a NumPy array")

    if subspaces.ndim != 3:
        raise ValueError(
            f"subspaces must be 3D (n_subspaces, dim, k). "
            f"Got shape {subspaces.shape}")

    distance_map = {
        "geodesic": distance_calculation.geo_distance,
        "chordal": distance_calculation.chordal_distance,
        "fubini_study": distance_calculation.Fubini_Study_distance,
        "smallest_angle": distance_calculation.small_angle,
    }

    if dist_type not in distance_map:
        raise ValueError(f"dist_type must be one of {list(distance_map.keys())}")

    dist_fn = distance_map[dist_type]

    n_subspaces = subspaces.shape[0]
    dist_matrix = np.zeros((n_subspaces, n_subspaces))

    for i in range(n_subspaces):
        for j in range(n_subspaces):
            dist_matrix[i, j] = dist_fn(subspaces[i],subspaces[j])

    return dist_matrix

#sbspces = build_subspaces_fast(X, 10)
#print(sbspces)

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

def MDS_SUBSPACE_EMBEDDING(X_all):
    # I wil finsih this tomorrow. Think about if hyou want it to be split into samples already, or to pass in all the data