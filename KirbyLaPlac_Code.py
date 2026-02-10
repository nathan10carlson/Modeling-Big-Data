import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import pandas as pd
from scipy.spatial.distance import squareform

# ===============================
# 1️⃣ Load your distance/dissimilarity matrix
# ===============================
# Replace this with your actual data loading
# e.g., DM = pd.read_excel('morse_dist.xlsx').to_numpy()
# Here we mimic MATLAB's squareform(dissimilarities)
# dissimilarities = ...
# DM = squareform(dissimilarities)

# For demonstration, let's make a dummy 10x10 distance matrix
skiing = pd.read_excel('data/skiing_dist.xlsx', index_col=0)
Dist = skiing.to_numpy(dtype=float)
labels = skiing.index.to_numpy()
n = Dist.shape[0]
DM = Dist

#DM = np.random.rand(10, 10)
#DM = (DM + DM.T)/2  # make symmetric
np.fill_diagonal(DM, 0)

n = DM.shape[0]

# ===============================
# 2️⃣ Initialize matrices
# ===============================
A = np.zeros((n, n))      # adjacency
W = np.zeros((n, n))      # weighted adjacency
D = np.zeros((n, n))      # degree for adjacency
DW = np.zeros((n, n))     # degree for weighted adjacency

T = 5000      # scaling for weights
K = 3         # number of nearest neighbors

# ===============================
# 3️⃣ Compute adjacency (K-nearest neighbors, symmetric)
# ===============================
# argsort to get nearest neighbors
sorted_idx = np.argsort(DM, axis=0)  # sort each column
for i in range(n):
    for j in range(n):
        if i != j:
            S1 = sorted_idx[1:K+1, i]  # nearest neighbors of i
            S2 = sorted_idx[1:K+1, j]  # nearest neighbors of j
            # connect if i in j's neighbors or j in i's neighbors
            if i in S2 or j in S1:
                A[i, j] = 1
            else:
                A[i, j] = 0

# ===============================
# 4️⃣ Compute weighted adjacency
# ===============================
for i in range(n):
    for j in range(n):
        if A[i, j] == 1:
            W[i, j] = np.exp(-DM[i, j]**2 / T)
        else:
            W[i, j] = 0

# ===============================
# 5️⃣ Compute degree matrices
# ===============================
D = np.diag(np.sum(A, axis=0))
DW = np.diag(np.sum(W, axis=0))

# ===============================
# 6️⃣ Compute Laplacians
# ===============================
L = D - A      # unweighted Laplacian
LW = DW - W    # weighted Laplacian

# ===============================
# 7️⃣ Solve generalized eigenproblems
# L v = lambda D v  and  LW v = lambda DW v
# ===============================
eigvals_L, eigvecs_L = eigh(L, D)
eigvals_W, eigvecs_W = eigh(LW, DW)

# sort eigenvalues and eigenvectors
idx_L = np.argsort(eigvals_L)
eigvals_L = eigvals_L[idx_L]
eigvecs_L = eigvecs_L[:, idx_L]

idx_W = np.argsort(eigvals_W)
eigvals_W = eigvals_W[idx_W]
eigvecs_W = eigvecs_W[:, idx_W]

# ===============================
# 8️⃣ 2D embedding (skip first eigenvector)
# ===============================
X = eigvecs_L[:, 1:3].T   # unweighted
X2 = eigvecs_W[:, 1:3].T  # weighted

# ===============================
# 9️⃣ Plot
# ===============================
plt.figure(figsize=(8, 6))
plt.plot(X[0, :], X[1, :], 'ro', markersize=13, label='adjacency A')
plt.plot(X2[0, :], X2[1, :], 'bv', markersize=13, label='weighted adjacency W')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.title('2D Laplacian Eigenmap (Unweighted vs Weighted)')
plt.legend()
plt.grid(True)
plt.show()