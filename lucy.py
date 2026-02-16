import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import loadmat
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# --- Load (or create) your data matrix X ---
# For example: X shape = (n_points, n_features)
# In your MATLAB script, DM is the dissimilarity:
# Here we start with data and compute distances.
# Replace the next line with your own data loading.

# load data
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/cat_dogs.mat")
## 4096 X 198
matrix = data['Y']
X = matrix.T
# X = np.random.rand(50, 10)  # example data


labels = [str(i) for i in range(X.shape[0])]

# --- Pairwise distance matrix (Euclidean) ---
dist_matrix = squareform(pdist(X, metric='euclidean'))

D = dist_matrix
print(D.shape)
n = len(D)
eeT = np.ones((n, n), dtype=int)
I = np.eye(n)

H = I - (1 / n) * eeT

B = -1 / 2 * H @ (D * D) @ H

evals, evecs = np.linalg.eigh(B)

idx = evals.argsort()[::-1]
sorted_evals = evals[idx]
sorted_evecs = evecs[:, idx]

Lambda = np.diag(sorted_evals, k=0)
V = sorted_evecs

# --- 2D embedding ---
d = 2
Lambda = np.diag(sorted_evals[:d])
V_tilde = (sorted_evecs[:, :d] @ np.sqrt(Lambda)).T

# --- Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(V_tilde[0, :], V_tilde[1, :], s=50, color="blue")

plt.scatter(V_tilde[0, :99], V_tilde[1, :99], c='red', label='Dogs')
plt.scatter(V_tilde[0, 99:], V_tilde[1, 99:], c='blue', label='Cats')
'''
for i, txt in enumerate(labels):
    plt.text(V_tilde[0, i] + 0.01, V_tilde[1, i] + 0.01, txt)

'''

plt.legend()
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Classical MDS")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot eigenvalues (Scree Plot) ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(sorted_evals) + 1), sorted_evals, marker='o')
plt.title("Eigenvalue Spectrum (Scree Plot)")
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.show()





