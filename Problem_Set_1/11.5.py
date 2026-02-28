import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# Number of nodes

Q_manual = np.array([
    [ 1,  1,  1,  0,  0, -1,  0,  0,  0,  0],
    [-1,  0,  0,  1,  1,  0,  0,  0, -1,  0],
    [ 0,  0,  0,  0, -1,  1,  1,  0,  0,  0],
    [ 0, -1,  0, -1,  0,  0,  0,  1,  0, -1],
    [ 0,  0,  0,  0,  0,  0, -1, -1,  1,  1],
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0]
], dtype=int)
print("\nIncidence matrix Q_manual:\n")
QQ_T = Q_manual @ Q_manual.T
print(QQ_T)

# computing node degree matrix (pulling diagonal of QQ_T)
Deg_A = np.diag(QQ_T)
print(Deg_A)
Node_Deg = np.diag(Deg_A)
print(Node_Deg)

L_A = QQ_T
print(L_A.shape)
eigvals_A, eigvecs_A = eigh(L_A, Node_Deg)

# Sorting evals for both cases
idx_A = np.argsort(eigvals_A)
eigvals_A, eigvecs_A = eigvals_A[idx_A], eigvecs_A[:, idx_A]
print('Checking smallest eigenvalues')
print(np.min(eigvals_A))

# 2D embeddings
embedding_A = eigvecs_A[:, 1:3]

plt.figure(figsize=(8,6))
plt.scatter(eigvecs_A[:,1], eigvecs_A[:,2])
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()

# ---- extract edges from incidence matrix ---- from CHATGPT
edges = []

num_nodes, num_edges = Q_manual.shape

for j in range(num_edges):
    nodes = np.where(Q_manual[:, j] != 0)[0]

    # only keep standard edges (2-node connections)
    if len(nodes) == 2:
        edges.append((nodes[0], nodes[1]))

print("Edges:", edges)

# ---- plotting with edges ----
plt.figure(figsize=(8, 6))

x = eigvecs_A[:, 1]
y = eigvecs_A[:, 2]

# draw edges
for i, j in edges:
    plt.plot([x[i], x[j]], [y[i], y[j]])

# draw nodes
plt.scatter(x, y)

# label nodes
for k in range(len(x)):
    plt.text(x[k] + 0.01, y[k] + 0.01, str(k + 1), fontsize=11)

plt.xlabel('component 1')
plt.ylabel('component 2')
plt.title('Spectral Embedding with Graph Edges')
plt.show()