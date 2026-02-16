from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import string

k = 2
T = 10000

# options
plot_mode = 'weighted'  # options: 'unweighted', 'weighted', 'both'
#plot_mode = 'both'  # options: 'unweighted', 'weighted', 'both'
#plot_mode = 'both'  # options: 'unweighted', 'weighted', 'both'


# Extract the distance matrix (as numpy array)
#matrix = data['dissMatrix']  # replace with your actual variable name
# Slice to get only the first 26 rows and columns (A-Z)
# Define the matrix
Dist = np.array([
    [0., 177, 177, 166, 188],
    [177, 0, 96, 79, 166],
    [177, 96, 0, 144, 177],
    [166, 79, 144, 0, 177],
    [188, 166, 177, 177, 0]
])
##Generating labesl
# Letters A-Z
import string
# Get labels A-E
labels = list(string.ascii_lowercase[:5])
print(labels)


# Convert to DataFrame, skipping the first row/column if they are just indices
# Assuming the first row and first column are index numbers
#Dist = pd.DataFrame(matrix_letters[0:, 0:])


n = Dist.shape[0]



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
    plt.scatter(embedding_W[:,0], embedding_W[:,1], color='blue', label='Weighted', s=60, marker='o')

# Add labels for clarity (use unweighted as reference)
for i, label in enumerate(labels):
    plot_shift = 0.5
    if plot_mode == 'unweighted' or plot_mode == 'both':
        plt.text(embedding_A[i,0]+plot_shift, embedding_A[i,1]+plot_shift, label, fontsize=9)
    if plot_mode == 'weighted' or plot_mode == 'both':

        plt.text(embedding_W[i,0]+plot_shift, embedding_W[i,1]+plot_shift, label, fontsize=9)

plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
if plot_mode == 'weighted' or plot_mode == 'both':
    plt.title(f'2D Laplacian Eigenmap ({k}-NN, T={T:,})')
else:
    plt.title(f'2D Laplacian Eigenmap ({k}-NN)')
plt.legend()
plt.grid(True)
plt.show()


