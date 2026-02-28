from scipy.io import loadmat
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

data = loadmat("/programs/MATH 532/data/cat_dogs.mat")
## 4096 X 198
print(data["Y"].shape)
data = data["Y"]
# adding labels!
n_samples = data.shape[1]  # 198
n_cats = n_samples // 2          # 99
n_dogs = n_samples - n_cats      # 99

labels = np.array([0]*n_cats + [1]*n_dogs)  # 0=cat, 1=dog
print(labels.shape)  # (198,)
print(labels[:10])   # first 10 labels

## To begin, we need a distance matrix. I am pulling one from the MDS jupyter ntoebook on canvas
def distmat(X, method='2norm'):
    n = X.shape[1]
    D = np.zeros((n, n))

    if method == '2norm':
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[:, i] - X[:, j])
    elif method == 'angle':
        for i in range(n - 1):
            for j in range(i + 1, n):
                x = X[:, i]
                y = X[:, j]
                D[i, j] = np.arccos(np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                # Alternatively, use D(i,j) = min(j-i, n+i-j) * 2 * np.pi / 7
    else:
        print('Unknown distance method')

    D += D.T
    return D

data = data.astype(np.float64)/255
Dist = distmat(data, method='2norm')

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
plt.title('Eigenvalues of Cat and Dogs Data')

# --- Subplot 2: Cumulative Energy ---
plt.subplot(1, 2, 2)
plt.plot(range(len(eigvals)), energy, marker='o', color='orange')
plt.xlabel('Index')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy of Eigenvalues for Cat and Dogs Dataset')
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

# Boolean masks
cats = labels == 0
dogs = labels == 1

# Plot separately so they get different colors
plt.scatter(V_tilde[0, cats], V_tilde[1, cats], marker='o')
plt.scatter(V_tilde[0, dogs], V_tilde[1, dogs], marker='x')

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D MDS")
plt.legend(["Cats", "Dogs"])
plt.grid(True)
plt.show()