from scipy.io import loadmat
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

Dist = np.array([[0,1,1,1],
                [1,0,1,1],
                [1,1,0,1],
                 [1,1,1,0]])

n = Dist.shape[0]


A = (-1/2) * (Dist * Dist )

n = Dist.shape[0]
H = np.eye(n) - (1. / n) * np.ones((n,1)) @ np.ones((n,1)).T

B = H @ A @ H

print(0.1>np.sum(B, axis=0))
print(0.1>np.sum(B, axis=1))
# if prints all true, then mean centering worked! :0

# ciompute eigen decomposition
eigvals, eigvecs = eigh(B)
#eigvals, eigvecs = np.linalg.eigh(B)


# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
tol = 1e-9
pos = eigvals > tol
print(pos)
eigvals = eigvals[pos]
eigvecs = eigvecs[:, pos]



# compute d, number of evals greater than 0
print('below')
d = 0
for i in range(len(eigvals)):
    if eigvals[i] > 0:
        d += 1
print(f'There are {d} positive-non-zero eigenvals')
print(len(eigvals))

# constructing eigen decomp
Lambda = np.diag(eigvals)
V = eigvecs.copy()

# Plotting eigenvaluse

# Compute cumulative energy
energy = np.cumsum(eigvals) / np.sum(eigvals)

# Create a figure with 2 subplots
plt.figure(figsize=(10, 5))

# --- Subplot 1: Eigenvalues ---
plt.subplot(1, 2, 1)
plt.plot(range(len(eigvals)), eigvals, marker='o')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
#plt.yscale('log')
plt.title('Eigenvalues for 4 Equidistant Points')

# --- Subplot 2: Cumulative Energy ---
plt.subplot(1, 2, 2)
plt.plot(range(len(eigvals)), energy, marker='o', color='orange')
plt.xlabel('Index')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy of Eigenvalues for 4 Equidistant Points')
plt.ylim([0, 1.05])  # ensure it goes slightly above 1

plt.tight_layout()
plt.show()

#print("Eigenvalues (descending):", eigvals)
#print("Lambda matrix:\n", Lambda)
#print("Eigenvectors shape:", eigvecs.shape)

# Verify first eigenpair to make sure i got this right
#v = eigvecs[:, 0]
#residual = B @ v - Lambda[0,0] * eigvecs[:, 0]
#print(residual)
#print("Residual norm (should be ~0):", np.linalg.norm(residual))

#Compute normalized eigenvtor matrix
#confirm = V @ Lambda @ V.T
d = 2

# compute configuration (columns are points)
Lambda_d = np.diag(eigvals[:d])
V_d = eigvecs[:, :d]
V_tilde = (V_d @ np.sqrt(Lambda_d)).T

print(V_tilde.shape)

plt.figure(figsize=(8,6))
plt.scatter(V_tilde[0,:], V_tilde[1,:], marker='o')
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D MDS")
plt.grid(True)
plt.show()

from mpl_toolkits.mplot3d import Axes3D



from scipy.spatial.distance import pdist, squareform

# Transpose so rows are points
X = V_tilde.T   # shape (4, d)

# Compute pairwise Euclidean distances
D_reconstructed = squareform(pdist(X))

print("Reconstructed distance matrix:")
print(np.round(D_reconstructed, 6))