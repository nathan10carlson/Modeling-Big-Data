import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import GSVD_goated_script as GSVD

# =========================
# 1. Load data
# =========================
data = loadmat("../data/trig_dataset_526x4.mat")
A = data["X"]
t = data["t"].flatten()

m, n = A.shape

print("A shape:", A.shape)
print(t.shape)
# =========================
# 2. Construct B = (1/sqrt(2)) dA
# =========================
shifted = np.roll(A, shift=1, axis=0)
B = (A - shifted) / np.sqrt(2)

print("B shape:", B.shape)

# =========================
# 3. GSVD of (A, B)
# =========================
U, V, C, S, G = GSVD.GSVD(A, B)

# =========================
# 4. Estimate M and T1
# =========================
M_est = G
T1_gsvd = U

print("Estimated M shape:", M_est.shape)

# =========================
# 5. SVD of A for comparison
# =========================
Ua, Sa, VaT = np.linalg.svd(A, full_matrices=False)

# =========================
# 6. Plot GSVD basis T1
# =========================
plt.figure(figsize=(8, 5))
for i in range(T1_gsvd.shape[1]):
    plt.plot(t, T1_gsvd[:, i], label=f"GSVD U{i}")

plt.title("GSVD Estimated Basis $T_1$")
plt.grid()
plt.legend()
plt.show()

# =========================
# 7. Plot SVD basis
# =========================
plt.figure(figsize=(8, 5))
for i in range(Ua.shape[1]):
    plt.plot(t, Ua[:, i], linestyle='--', label=f"SVD U{i}")

plt.title("SVD Basis of A")
plt.grid()
plt.legend()
plt.show()

# =========================
# 8. Compare GSVD vs SVD directly
# =========================
plt.figure(figsize=(8, 5))

for i in range(min(4, n)):
    plt.plot(t, T1_gsvd[:, i], label=f"GSVD U{i}")

for i in range(min(4, n)):
    plt.plot(t, Ua[:, i], linestyle='--', label=f"SVD U{i}")

plt.title("GSVD vs SVD Basis Comparison")
plt.grid()
plt.legend()
plt.show()

# =========================
# 9. Plot generalized singular values
# =========================
c = np.diag(C)
s = np.diag(S)

plt.figure(figsize=(6, 4))
plt.plot(c**2, label="c^2")
plt.plot(s**2, label="s^2")
plt.title("GSVD Singular Value Structure")
plt.legend()
plt.grid()
plt.show()

# =========================
# 10. Reconstruction check
# =========================
A_rec = U @ C @ G.T
B_rec = V @ S @ G.T

print("Reconstruction error A:", np.linalg.norm(A - A_rec))
print("Reconstruction error B:", np.linalg.norm(B - B_rec))

# =========================
# 11. Output estimate of M
# =========================
print("M estimate (first 5 rows):")
print(M_est[:5])