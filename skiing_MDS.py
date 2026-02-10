import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# importing dataset
skiing = pd.read_excel('data/skiing_dist.xlsx', index_col=0)
labels = skiing.index.to_numpy()
print(skiing.head())   # first 5 rows
print(skiing.shape)    # (rows, columns)
print(skiing.columns)  # column names

# distance matrix
Dist = skiing.to_numpy(dtype=float)
#print(Dist)
#print(Dist - Dist.T)

A = (-1/2) * (Dist * Dist )

n = Dist.shape[0]
H = np.eye(n) - (1. / n) * np.ones((n,1)) @ np.ones((n,1)).T

B = H @ A @ H

print(0.1>np.sum(B, axis=0))
print(0.1>np.sum(B, axis=1))
# if prints all true, then mean centering worked! :0

# ciompute eigen decomposition
eigvals, eigvecs = eigh(B)

# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
# compute d, number of evals greater than 0
print('below')
d = 0
for i in range(len(eigvals)):
    if eigvals[i] > 0:
        d += 1
print(f'There are {d} positive-non-zero eigenvals')

# constructing eigen decomp
Lambda = np.diag(eigvals)
V = eigvecs.copy()

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
d=2

# compute V_tilde .... configuration is columns
V_tilde = (V[:,0:d] @ Lambda[0:d,0:d]**.50).T
print(V_tilde.shape)

plt.figure(figsize=(8,6))
plt.scatter(V_tilde[0,:], V_tilde[1,:], color='blue')

for i, label in enumerate(labels):
    plt.text(V_tilde[0, i] + 0.01, V_tilde[1, i] + 0.01, label, fontsize=9)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D MDS")
plt.grid(True)
plt.show()