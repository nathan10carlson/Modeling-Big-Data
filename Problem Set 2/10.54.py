import numpy as np
import scipy.io
import matplotlib.pyplot as plt


# ==============================
# LOAD DATA
# ==============================
file_path = "/Users/nathancarlson/Desktop/programs/MATH 532/Problem Set 2/Kingrynormalized.mat"

mat = scipy.io.loadmat(file_path)

# Inspect keys to find the variable name
print(mat.keys())

Kingrynorm = mat["Kingrynorm"]


# Samples are columns
print("Data shape:", Kingrynorm.shape)

# ==============================
# SVD
# ==============================
U, S, VT = np.linalg.svd(Kingrynorm, full_matrices=False)

print("U shape:", U.shape)
print("S shape:", S.shape)
print("VT shape:", VT.shape)

print('Singular values:', S)

plt.figure()
plt.semilogy(S)  # log scale is standard for singular values
plt.title("Singular Values of Kingrynorm")
plt.xlabel("Index")
plt.ylabel("Singular Value (log scale)")
plt.grid(True)
plt.show()

# Energy computation

sing_sum = np.sum(S**2)
cum_sum = np.cumsum(S**2)
energy = cum_sum / sing_sum
print(f"Cumulative energy in 3-D is: {energy[2]}")


# Plot energy curve

k_vals = np.arange(1, len(energy) + 1)

plt.figure()
plt.plot(k_vals, energy, marker='o')
plt.axhline(0.95, color='r', linestyle='--')
plt.title("Energy of Kingrynorm")
plt.xlabel("Number of singular values (k)")
plt.ylabel("Cumulative energy $E_k$")
plt.grid(True)
plt.show()

for i in range(len(energy)):
    if energy[i] >= 0.95:
        print(f'{i} singular values needed to achieve 95% energy.')
        break

# rank k approxiamtion
k = 3

U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

rank_k_approx = U_k @ S_k @ VT_k

print("Rank-k approximation shape:", rank_k_approx.shape)


plt.figure(figsize=(8, 6))
plt.imshow(rank_k_approx, aspect='auto', cmap='viridis')
plt.title(f"Rank-{k} Approximation Heatmap")
plt.xlabel("Samples")
plt.ylabel("Genes")
plt.colorbar(label="Expression level")
plt.savefig(f"rank{k}_heatmap.png", bbox_inches='tight')
plt.close()

# Projection onto first 3 components
proj = U[:, :3].T @ Kingrynorm
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(proj[0, :], proj[1, :], proj[2, :])

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
plt.title("3D PCA Projection")
plt.show()

Schu4_lung = Kingrynorm[:, 6:30]
LVS_lung   = Kingrynorm[:, 30:54]
Schu4_spleen = Kingrynorm[:, 60:84]
LVS_spleen   = Kingrynorm[:, 84:108]


def compute_energy_curve(X):
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    sing_sum = np.sum(S ** 2)
    cum_sum = np.cumsum(S ** 2)
    energy = cum_sum / sing_sum

    return energy

energy_schu4_lung = compute_energy_curve(Schu4_lung)
energy_lvs_lung   = compute_energy_curve(LVS_lung)
energy_schu4_spleen = compute_energy_curve(Schu4_spleen)
energy_lvs_spleen   = compute_energy_curve(LVS_spleen)

k_vals = np.arange(1, len(energy_schu4_lung) + 1)

plt.figure(figsize=(8,6))

plt.plot(k_vals, energy_schu4_lung, label="Schu4 Lung")
plt.plot(k_vals, energy_lvs_lung, label="LVS Lung")
plt.plot(k_vals, energy_schu4_spleen, label="Schu4 Spleen")
plt.plot(k_vals, energy_lvs_spleen, label="LVS Spleen")

plt.axhline(0.95, linestyle='--')

plt.title("Energy Curves $E_k$ for Different Conditions")
plt.xlabel("k (number of singular values)")
plt.ylabel("Cumulative Energy $E_k$")
plt.legend()
plt.grid(True)
plt.show()


def pca_top3(X):
    # SVD
    U, S, VT = np.linalg.svd(X, full_matrices=False)

    # Project data onto first 3 principal components
    proj = U[:, :3].T @ X  # shape: (3, n_samples)

    return proj, U, S, VT

proj_schu4_lung, U1, S1, VT1 = pca_top3(Schu4_lung)
proj_lvs_lung, U2, S2, VT2 = pca_top3(LVS_lung)
proj_schu4_spleen, U3, S3, VT3 = pca_top3(Schu4_spleen)
proj_lvs_spleen, U4, S4, VT4 = pca_top3(LVS_spleen)

from mpl_toolkits.mplot3d import Axes3D


def plot_3d(proj, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(proj[0, :], proj[1, :], proj[2, :])

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()

plot_3d(proj_schu4_lung, "Schu4 Lung PCA (Top 3)")
plot_3d(proj_lvs_lung, "LVS Lung PCA (Top 3)")
plot_3d(proj_schu4_spleen, "Schu4 Spleen PCA (Top 3)")
plot_3d(proj_lvs_spleen, "LVS Spleen PCA (Top 3)")