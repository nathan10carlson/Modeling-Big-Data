import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# options
#plot_mode = 'unweighted'  # options: 'unweighted', 'weighted', 'both'
plot_mode = 'both'  # options: 'unweighted', 'weighted', 'both'
#plot_mode = 'both'  # options: 'unweighted', 'weighted', 'both'
# Load dataset
skiing = pd.read_excel('data/skiing_dist.xlsx', index_col=0)
Dist = skiing.to_numpy(dtype=float)
labels = skiing.index.to_numpy()
print(Dist)
n = Dist.shape[0]

k = 5
T = 5000

# Knn
neighbors = np.argsort(Dist, axis=1)[:, 1:k+1]
A = np.zeros((n, n))
for i in range(n):
    A[i, neighbors[i]] = 1
A = np.maximum(A, A.T)
Deg_A = np.diag(np.sum(A, axis=0))

# weighted adjacency
W = np.exp(-Dist**2 / T) * A
Deg_W = np.diag(np.sum(W, axis=0))

# Laplac
L_A = Deg_A - A
print(L_A)
print('this is L_A')
L_W = Deg_W - W

# genearlaized eigen decomp
eigvals_A, eigvecs_A = eigh(L_A, Deg_A)
eigvals_W, eigvecs_W = eigh(L_W, Deg_W)

# Sorting evals for both cases
idx_A = np.argsort(eigvals_A)
eigvals_A, eigvecs_A = eigvals_A[idx_A], eigvecs_A[:, idx_A]
idx_W = np.argsort(eigvals_W)
eigvals_W, eigvecs_W = eigvals_W[idx_W], eigvecs_W[:, idx_W]
print('Checking smallest eigenvalues')
print(np.min(eigvals_A), np.min(eigvals_W))

# 2D embeddings
embedding_A = eigvecs_A[:, 1:3]
embedding_W = eigvecs_W[:, 1:3]

# Plotting (thanks CHATGPT)
plt.figure(figsize=(8,6))

if plot_mode == 'unweighted' or plot_mode == 'both':
    plt.scatter(embedding_A[:,0], embedding_A[:,1], color='red', label='Unweighted', s=60)
if plot_mode == 'weighted' or plot_mode == 'both':
    plt.scatter(embedding_W[:,0], embedding_W[:,1], color='blue', label='Weighted', s=60, marker='v')

# Add labels for clarity (use unweighted as reference)
for i, label in enumerate(labels):
    if plot_mode == 'unweighted' or plot_mode == 'both':
        plt.text(embedding_A[i,0]+0.01, embedding_A[i,1]+0.01, label, fontsize=9)
    if plot_mode == 'weighted' or plot_mode == 'both':
        plt.text(embedding_W[i,0]+0.01, embedding_W[i,1]+0.01, label, fontsize=9)

plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
plt.legend()
plt.grid(True)
plt.show()