import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# ---------------- Example Graph: K_4 ----------------
D = np.array([[3,0,0,0],
              [0,3,0,0],
              [0,0,3,0],
              [0,0,0,3]])

A = np.array([[0,1,1,1],
              [1,0,1,1],
              [1,1,0,1],
              [1,1,1,0]])

# Laplacian
L = D - A

# Solve generalized eigenproblem L u = lambda D u
eigvals, U = eigh(L, D)

# ---------------- 2D Embedding ----------------
X_2d = U[:, 1:3].T

# ---------------- 2D Plot with edges ----------------
plt.figure(figsize=(6,6))
plt.scatter(X_2d[0], X_2d[1], color='red', s=80, label='Vertices')

# Draw edges
n = X_2d.shape[1]
for i in range(n):
    for j in range(i+1, n):
        if A[i,j] != 0:
            plt.plot([X_2d[0,i], X_2d[0,j]], [X_2d[1,i], X_2d[1,j]], 'k-', alpha=0.5)

# Add vertex labels
for i in range(n):
    plt.text(X_2d[0,i]+0.02, X_2d[1,i]+0.02, f'v{i+1}', fontsize=12)

plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel('Eigenvector 2')
plt.ylabel('Eigenvector 3')
plt.title('2D Laplacian Eigenmap with edges')
plt.grid(True)
plt.legend()
plt.show()

# ---------------- 3D Embedding ----------------
X_3d = U[:, 1:4].T
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_3d[0], X_3d[1], X_3d[2], color='blue', s=80)

# Draw edges in 3D
for i in range(n):
    for j in range(i+1, n):
        if A[i,j] != 0:
            ax.plot([X_3d[0,i], X_3d[0,j]],
                    [X_3d[1,i], X_3d[1,j]],
                    [X_3d[2,i], X_3d[2,j]],
                    'k-', alpha=0.5)

# Add vertex labels
for i in range(n):
    ax.text(X_3d[0,i]+0.02, X_3d[1,i]+0.02, X_3d[2,i]+0.02, f'v{i+1}', fontsize=12)

ax.set_xlabel('Eigenvector 2')
ax.set_ylabel('Eigenvector 3')
ax.set_zlabel('Eigenvector 4')
ax.set_title('3D Laplacian Eigenmap with edges')
plt.show()

