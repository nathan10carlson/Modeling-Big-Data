# converted to python code with CGPT
import numpy as np
from scipy.spatial.distance import squareform
from scipy.linalg import eigh
import matplotlib.pyplot as plt


n = DM.shape[0]

A = np.zeros((n, n))   # adjacency (unweighted)
W = np.zeros((n, n))   # weighted adjacency
D = np.zeros((n, n))   # degree matrix (unweighted)
DW = np.zeros((n, n))  # degree matrix (weighted)

T = 1e8     # heat kernel bandwidth
K = 9       # number of nearest neighbors

# Sort each column: II[:, i] gives indices of neighbors of i
II = np.argsort(DM, axis=0)

for i in range(n):
    for j in range(n):
        if i != j:
            S1 = II[1:K+1, i]  # neighbors of i
            S2 = II[1:K+1, j]  # neighbors of j

            Q = i in S2        # i is neighbor of j?
            P = j in S1        # j is neighbor of i?

            if Q or P:
                A[i, j] = 1
            else:
                A[i, j] = 0


for i in range(n):
    for j in range(n):
        if A[i, j] == 1:
            W[i, j] = np.exp(-DM[i, j]**2 / T)
        else:
            W[i, j] = 0.0

for i in range(n):
    D[i, i]  = np.sum(A[:, i])
    DW[i, i] = np.sum(W[:, i])

L  = D - A      # unweighted Laplacian
LW = DW - W     # weighted Laplacian

eigvals, U = eigh(L, D)
eigvals_w, UW = eigh(LW, DW)

X  = U[:, 1:3].T
X2 = UW[:, 1:3].T


plt.figure()

plt.plot(X[0], X[1], 'ro', markersize=10, label='adjacency A')
plt.plot(X2[0], X2[1], 'bv', markersize=10, label='weighted adjacency W')

plt.axhline(0)
plt.axvline(0)

plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.legend()
plt.show()
