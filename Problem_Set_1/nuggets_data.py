import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# -----------------------------
# 1️⃣ Load and process data
# -----------------------------
raw_data = [
    ["A. GORDON F",23,642,25.3,8.7,17.0,50.9,2.6,6.5,40.0,5.4,6.8,78.9,1.9,7.0,8.8,3.6,1.5,1.0,0.2,2.4,11.8],
    ["B. BROWN G",55,1371,12.0,4.6,10.0,46.2,0.9,2.6,35.2,1.9,2.4,77.1,1.9,4.6,6.4,3.4,1.6,1.7,0.4,3.9,-2.0],
    ["C. JOHNSON F",31,942,15.1,5.3,11.3,47.0,2.5,5.7,43.7,2.0,2.5,81.0,0.9,3.9,4.8,2.9,1.2,0.9,0.3,3.1,8.7],
    ["C. BRAUN G",19,576,13.1,5.0,10.4,48.0,0.8,3.1,25.0,2.4,3.1,77.3,2.0,3.7,5.7,3.7,1.5,0.7,0.5,3.0,8.5],
    ["C. JONES G",6,31,11.7,5.2,18.2,28.6,1.3,11.7,11.1,0.0,0.0,0.0,1.3,5.2,6.5,3.9,1.3,0.0,0.0,3.9,5.2],
    ["D. HOLMES II F",15,176,13.2,4.3,9.8,44.2,3.2,7.7,41.2,1.4,1.8,75.0,1.1,4.3,5.5,3.4,1.1,0.0,0.9,3.6,-5.5],
    ["H. TYSON F",21,162,11.4,3.5,12.8,26.9,1.7,8.2,21.2,2.7,3.2,84.6,1.0,7.9,8.9,4.2,1.2,0.5,0.2,3.2,-5.9],
    ["J. PICKETT G",39,688,12.3,4.7,11.3,41.5,2.3,6.2,37.4,0.6,0.6,90.9,1.0,4.9,5.9,5.8,1.6,0.8,0.2,2.2,-0.9],
    ["J. MURRAY G",50,1780,28.9,10.2,21.1,48.5,3.6,8.4,42.5,4.9,5.6,88.7,0.5,4.4,4.9,8.5,2.6,1.1,0.4,1.8,5.1],
    ["J. VALANČIŪNAS C",43,627,25.0,10.0,17.4,57.7,0.3,1.1,27.8,4.7,6.1,76.0,4.5,9.9,14.4,3.5,3.3,0.6,1.8,6.2,-3.4],
    ["J. STRAWTHER G",32,441,18.3,6.3,14.1,45.2,2.0,6.3,31.9,3.6,4.6,78.4,0.5,5.5,6.0,2.8,1.7,1.3,0.2,3.4,-1.4],
    ["N. JOKIĆ C",39,1338,33.5,11.8,19.9,59.0,2.3,5.4,42.0,7.7,9.2,84.0,3.6,10.7,14.4,12.5,4.3,1.6,0.9,3.1,10.3],
    ["P. WATSON G",49,1502,19.5,7.0,14.2,49.6,2.0,4.8,41.7,3.4,4.7,72.7,1.2,5.2,6.4,2.6,2.3,1.3,1.5,3.2,4.7],
    ["S. JONES F",46,1086,10.1,3.6,7.2,50.5,1.8,4.3,41.4,1.0,1.7,62.2,1.9,3.4,5.3,1.4,1.0,1.5,0.7,4.8,-0.7],
    ["T. H. JR. G",54,1487,20.5,6.6,14.6,45.3,4.2,10.2,40.9,3.1,3.6,85.9,0.3,3.6,3.8,1.9,0.7,0.8,0.2,1.8,4.0],
    ["Z. NNAJI F",39,500,13.1,4.8,9.8,48.8,0.9,3.2,27.5,2.6,3.2,82.5,2.3,6.3,8.6,1.7,1.7,1.0,1.6,4.2,-7.8]
]

# Split NAME and POSITION
processed_data = []
for row in raw_data:
    name_pos = row[0].split()
    position = name_pos[-1]
    name = " ".join(name_pos[:-1])
    processed_data.append([name, position] + row[1:])

# Column names
columns = [
    "NAME","POSITION","GP","MIN","PTS","FGM","FGA","FG%","3PM","3PA","3P%",
    "FTM","FTA","FT%","OREB","DREB","REB","AST","TOV","STL","BLK","PF","+/-"
]

# Create DataFrame
df = pd.DataFrame(processed_data, columns=columns)
df = df.drop(index=4)

# Drop non-numeric columns for MDS
data_numeric = df.drop(columns=["NAME","POSITION","GP","MIN"]).to_numpy()

# Keep names for annotation
labels = df["NAME"].values

# -----------------------------
# 2️⃣ Distance matrix (Euclidean)
# -----------------------------
def distmat(X, method='2norm'):
    n = X.shape[0]  # rows = points
    D = np.zeros((n, n))
    if method == '2norm':
        for i in range(n-1):
            for j in range(i+1, n):
                D[i,j] = np.linalg.norm(X[i,:] - X[j,:])
    elif method == 'angle':
        for i in range(n-1):
            for j in range(i+1, n):
                x = X[i,:]
                y = X[j,:]
                D[i,j] = np.arccos(np.dot(x, y) / (np.linalg.norm(x)*np.linalg.norm(y)))
    D += D.T
    return D

Dist = distmat(data_numeric, method='2norm')

# -----------------------------
# 3️⃣ Classical MDS
# -----------------------------
n = Dist.shape[0]
A = -0.5 * (Dist ** 2)
H = np.eye(n) - np.ones((n,n))/n
B = H @ A @ H

eigvals, eigvecs = eigh(B)
idx = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# Keep only positive eigenvalues
pos = eigvals > 1e-9
eigvals = eigvals[pos]
eigvecs = eigvecs[:, pos]

# Compute 2D configuration
d = 2
Lambda_d = np.diag(eigvals[:d])
V_d = eigvecs[:, :d]
V_tilde = (V_d @ np.sqrt(Lambda_d)).T  # columns = points

# -----------------------------
# 4️⃣ Plot eigenvalues + energy
# -----------------------------
energy = np.cumsum(eigvals)/np.sum(eigvals)

plt.figure(figsize=(12,5))

# Eigenvalues
plt.subplot(1,2,1)
plt.plot(range(len(eigvals)), eigvals, marker='o')
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Eigenvalue')
plt.title('Eigenvalues (log scale)')

# Cumulative energy
plt.subplot(1,2,2)
plt.plot(range(len(eigvals)), energy, marker='o', color='orange')
plt.xlabel('Index')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy of Eigenvalues')
plt.ylim([0,1.05])

plt.tight_layout()
plt.show()

# -----------------------------
# 5️⃣ Plot 2D MDS
# -----------------------------
plt.figure(figsize=(10,8))
plt.scatter(V_tilde[0,:], V_tilde[1,:], c='blue')

# Annotate each player
for i, name in enumerate(labels):
    offset = 0.5
    plt.text(V_tilde[0,i]+offset, V_tilde[1,i]+offset, name, fontsize=8)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D MDS of Nuggets Players Stats (per 40 min)")
plt.grid(True)
plt.show()

## Implementing LAPLAC

n = Dist.shape[0]

k = 5
T = np.median(Dist)**2
#T = 5000

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
L_W = Deg_W - W
print(np.linalg.norm(L_W - L_W.T))
print('norm was')
print("Min degree:", np.min(np.diag(Deg_A)))
print("Zero degree nodes:", np.sum(np.diag(Deg_A) == 0))
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

# -----------------------------
# 6️⃣ Laplacian Eigenmaps Plot (Weighted/Unweighted/Both) – Nuggets Colors
# -----------------------------
plot_mode = 'weighted'  # options: 'unweighted', 'weighted', 'both'

# Define Nuggets colors
color_unweighted = '#FDB927'  # gold
color_weighted   = '#1D428A'  # blue

plt.figure(figsize=(10,8))

if plot_mode in ['unweighted', 'both']:
    plt.scatter(embedding_A[:,0], embedding_A[:,1], c=color_unweighted, marker='o', label='Unweighted')

if plot_mode in ['weighted', 'both']:
    plt.scatter(embedding_W[:,0], embedding_W[:,1], c=color_weighted, marker='x', label='Weighted')

# Annotate each player
for i, name in enumerate(labels):
    offset = 0.01
    if plot_mode in ['weighted', 'both']:
        plt.text(embedding_W[i,0]+offset, embedding_W[i,1]+offset, name, fontsize=8)
    if plot_mode in ['unweighted', 'both']:
        plt.text(embedding_A[i,0]+offset, embedding_A[i,1]+offset, name, fontsize=8)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title(f'2D Laplacian Eigenmaps of Nuggets Players Stats ({k}-NN, {plot_mode})')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 7️⃣ Fiedler Vector Embedding (1D) – Nuggets Colors
# -----------------------------
plt.figure(figsize=(10,6))

# Fiedler vectors: 2nd smallest eigenvector
fiedler_A = eigvecs_A[:,1]
fiedler_W = eigvecs_W[:,1]

if plot_mode in ['unweighted', 'both']:
    plt.scatter(range(len(fiedler_A)), fiedler_A, c=color_unweighted, marker='o', label='Unweighted')
if plot_mode in ['weighted', 'both']:
    plt.scatter(range(len(fiedler_W)), fiedler_W, c=color_weighted, marker='x', label='Weighted')

# Annotate each player
for i, name in enumerate(labels):
    offset = 0.02
    if plot_mode in ['weighted', 'both']:
        plt.text(i, fiedler_W[i]+offset, name, fontsize=8, rotation=45, ha='right')
    if plot_mode in ['unweighted', 'both']:
        plt.text(i, fiedler_A[i]+offset, name, fontsize=8, rotation=45, ha='right')

plt.xlabel("Player Index")
plt.ylabel("Fiedler Embedding (2nd smallest eigenvector)")
plt.title(f'Fiedler Vector Embeddings of Nuggets Players Stats ({k}-NN, {plot_mode})')
plt.legend()
plt.grid(True)
plt.show()