import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# importing dataset
skiing = pd.read_excel('skiing_dist.xlsx', index_col=0)

print(skiing.head())   # first 5 rows
print(skiing.shape)    # (rows, columns)
print(skiing.columns)  # column names

# distance matrix
Dist = skiing.to_numpy(dtype=float)

# labels (ski resort names in this case!)
labels = skiing.index.to_numpy()

print("Shape:", Dist.shape)

k = 3

# sort distances row-wise
neighbors = np.argsort(Dist, axis=1)[:, 1:k+1]
# neighbhors gives the indices

print("\nK nearest neighbors:")
for i, nbrs in enumerate(neighbors):
    print(labels[i], "->", labels[nbrs])

n = Dist.shape[0]

# building adjacency matrix
A = np.zeros((n, n))

for i in range(n):
    A[i, neighbors[i,:]] = 1

# make symmetric (add relations both way)
A = np.maximum(A, A.T)

# creating Degree matrix
row_sum = np.sum(A, axis=1)
Deg = np.diag(row_sum)

def compute_weight_adj(A, Dist, T=5000):
    W = np.exp(-1 *Dist**2 / T ) * A
    return W
W = compute_weight_adj(A, Dist)
print(W)
# computing weight LaPlacian
L = Deg - W
print(L)

# get evals, evecs
eigvals, eigvecs = eigh(L, Deg)

#  Sort eigenvalues and eigenvectors in ascending order
idx = np.argsort(eigvals)          # indices that would sort eigvals
eigvals = eigvals[idx]              # sorted eigenvalues
eigvecs = eigvecs[:, idx]
print(eigvals)

# 2D embedding
embedding_2D = eigvecs[:, 1:3]

plt.figure(figsize=(8,6))
plt.scatter(embedding_2D[:, 0], embedding_2D[:, 1], color='blue')

# add labels
for i, label in enumerate(labels):
    plt.text(embedding_2D[i, 0] + 0.01,  # small offset
             embedding_2D[i, 1] + 0.01,
             label,
             fontsize=9)

plt.xlabel("Eigenvector 2")
plt.ylabel("Eigenvector 3")
plt.title(f'2D Laplacian Eigenmap of Ski Resorts ({k} neighbors)')
plt.grid(True)
plt.show()
print(eigvecs.shape)

