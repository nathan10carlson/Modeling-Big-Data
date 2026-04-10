import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import qr, svd

# =========================
# 1. Load data
# =========================
data = loadmat("../data/trig_dataset_526x4.mat")
X = data["X"]
t = data["t"].flatten()

m, n = X.shape

# =========================
# 2. Build S1 and QR
# =========================
S1 = X[:, ::-1] @ np.diag([4, 3, 2, 1])
T1, R1 = qr(S1, mode='economic')

print("T1^T T1:\n", T1.T @ T1)

# =========================
# 3. Build T2
# =========================
XS = np.zeros_like(X)
XS[0, :] = X[-1, :]
XS[1:, :] = X[:-1, :]

T2 = (X - XS) / np.sqrt(2)

Dtrue = T2.T @ T2
print("Dtrue:\n", Dtrue)

# =========================
# 4. Mixing (set MIX yourself!)
# =========================
MIX = np.random.rand(4, 4)

A = T1 @ MIX
B = T2 @ MIX

m, n = A.shape
p, n = B.shape

# =========================
# 5. Stack and SVD
# =========================
M = np.vstack([A, B])
k = np.linalg.matrix_rank(M)
l, n = M.shape

Qh, SigmaH, Wt = svd(M, full_matrices=False)
W = Wt.T

W1 = W[:, :k]
W2 = W[:, k:] if k < n else np.empty((n, 0))

Sigmak = np.diag(SigmaH[:k])

Q11 = Qh[:m, :k]
Q21 = Qh[m:l, :k]

# =========================
# 6. CS decomposition
# =========================
#
from GSVD_goated_script import CSAlgorithm

U, V, C, S, Xcs = CSAlgorithm(Q11, Q21)

# =========================
# 7. Build G
# =========================
G = np.hstack([W1 @ Sigmak @ Xcs, W2])

# =========================
# 8. Extract diagonals
# =========================
c = np.diag(C)

if S.shape[0] == S.shape[1]:
    s = np.diag(S)
else:
    s = np.diag(S[-len(c):, :])

# =========================
# 9. Plot c^2 and s^2
# =========================
plt.figure()
plt.plot(c**2, label='c^2')
plt.plot(s**2, label='s^2')
plt.legend()
plt.grid()
plt.show()

# =========================
# 10. Errors
# =========================
errA = np.linalg.norm(A - U @ C @ Xcs.T @ Sigmak @ W1.T)
errB = np.linalg.norm(B - V @ S @ Xcs.T @ Sigmak @ W1.T)

print("errA:", errA)
print("errB:", errB)

# =========================
# 11. Block form check
# =========================
indc = C.shape[0]
inds = S.shape[0]

Oc = np.zeros((indc, n - k))
Os = np.zeros((inds, n - k))

errAd = np.linalg.norm(A - U @ np.hstack([C, Oc]) @ G.T)
errBd = np.linalg.norm(B - V @ np.hstack([S, Os]) @ G.T)

print("errAd:", errAd)
print("errBd:", errBd)

# =========================
# 12. Recover MIX
# =========================
MIXapp = C @ G.T
print("MIX true:\n", MIX)
print("MIX approx:\n", MIXapp)

Dapp = np.linalg.inv(C) @ (S @ S) @ np.linalg.inv(C)
print("Dapp:\n", Dapp)

# =========================
# Heatmap of T1
# =========================
plt.figure()

plt.imshow(
    T1,
    aspect='auto',
    origin='lower'
)

plt.xlabel("Signal index")
plt.ylabel("Time index t")
plt.title("T1")
plt.colorbar(label="Value")

plt.savefig("T1_heatmap.png", dpi=600)
plt.show()

# =========================
# 14. Plot A
# =========================
plt.figure()
for i in range(4):
    plt.plot(t, A[:, i], linewidth=2)

plt.xlabel("t")
plt.ylabel("Value")
plt.grid()
plt.savefig("A.png", dpi=600)

# =========================
# 15. Plot U
# =========================
plt.figure()
for i in range(4):
    plt.plot(t, U[:, i], linewidth=2)

plt.xlabel("t")
plt.ylabel("Value")
plt.grid()
plt.savefig("U.png", dpi=600)

Ua, Sa, Vta = svd(A, full_matrices=False)

import matplotlib.pyplot as plt

U_to_plot = U[:, :4]

plt.figure(figsize=(14, 6))

for i in range(4):

    # =====================
    # GSVD: T1
    # =====================
    plt.subplot(2, 4, i + 1)
    plt.plot(t, T1[:, i], linewidth=2)
    plt.title(f"T1 col {i+1}")
    plt.grid()
    plt.xticks([])
    plt.yticks([])

    # =====================
    # SVD: U
    # =====================
    plt.subplot(2, 4, i + 5)
    plt.plot(t, U_to_plot[:, i], linewidth=2, linestyle='--')
    plt.title(f"U col {i+1}")
    plt.grid()
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig("GSVD_vs_SVD_T1_U.png", dpi=600)
plt.show()

##
plt.figure()

plt.imshow(
    U_to_plot,
    aspect='auto',
    origin='lower'
)

plt.xlabel("Signal index")
plt.ylabel("Time index t")
plt.title("U[:,:4]")
plt.colorbar(label="Value")

plt.savefig("U_heatmap.png", dpi=600)
plt.show()