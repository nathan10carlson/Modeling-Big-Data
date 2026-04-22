import numpy as np
import matplotlib.pyplot as plt
import GSVD_goated_script as GSVD
from scipy.io import loadmat
import os

# =========================
# Output directory for images
# =========================
OUTPUT_DIR = "gsvd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load ground truth
# =========================
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/indian_pines_gt.mat")
gt = data["indian_pines_gt"]

plt.figure(figsize=(6, 6))
plt.imshow(gt, cmap='jet')
plt.title("Indian Pines Ground Truth")
plt.colorbar()
plt.axis('off')

plt.savefig(os.path.join(OUTPUT_DIR, "ground_truth.png"), dpi=300, bbox_inches='tight')
plt.show()


# =========================
# Load hyperspectral cube
# =========================
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/indian_pines_corrected.mat")
to_analyze = data["indian_pines_corrected"]

print(data.keys())
print(to_analyze.shape)

flattened_data = to_analyze.reshape(145 * 145, 200)

band = flattened_data  # shape (145,145)

# =========================
# Noise estimation
# =========================
shifted = np.roll(band, shift=1, axis=0)
N = (band - shifted) / np.sqrt(2)

plt.figure(figsize=(6, 6))
plt.imshow(N, cmap='gray')
plt.title("Estimated Noise N = (1/sqrt(2)) dX")
plt.axis('off')
plt.colorbar()

plt.savefig(os.path.join(OUTPUT_DIR, "noise_estimate.png"), dpi=300, bbox_inches='tight')
plt.show()


# =========================
# GSVD
# =========================
U, V, C, S, G = GSVD.GSVD(band, N)

c = np.diag(C)
s = np.diag(S)
i_vals = np.arange(len(c))

plt.figure(figsize=(8, 5))
plt.plot(i_vals, c, marker='o', label='c_i (signal)')
plt.plot(i_vals, s, marker='o', label='s_i (noise)')
plt.xlabel("Index i")
plt.ylabel("Value")
plt.title("GSVD Singular Value Pairs vs Index")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "gsvd_singular_values_full.png"),
            dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(8, 5))
plt.plot(i_vals, c, marker='o', label='c_i (signal)')
plt.plot(i_vals, s, marker='o', label='s_i (noise)')
plt.xlabel("Index i")
plt.xlim(0, 5)
plt.ylabel("Value")
plt.title("GSVD Singular Value Pairs (Zoomed)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTPUT_DIR, "gsvd_singular_values_zoom.png"),
            dpi=300, bbox_inches='tight')
plt.show()


# =========================
# Rank selection
# =========================
i_star = np.argmin(np.abs(c - s))
print("optimal i:", i_star)

i = i_star

rank_i_approx = U[:, :i] @ C[:i, :i] @ G[:, :i].T

img = rank_i_approx

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title(rf"Rank {i} Approximation")
plt.axis('off')
plt.colorbar()

plt.savefig(os.path.join(OUTPUT_DIR, f"rank_{i}_approximation.png"),
            dpi=300, bbox_inches='tight')
plt.show()