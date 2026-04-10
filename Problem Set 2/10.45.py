import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ==============================
# LOAD IMAGE
# ==============================
image_path = "/Users/nathancarlson/Desktop/coin.jpeg"
img = Image.open(image_path).convert("L")  # grayscale
A = np.array(img, dtype=float) / 255.0

# ==============================
# SVD
# ==============================
U, S, VT = np.linalg.svd(A, full_matrices=False)

# ==============================
# (a) Original image
# ==============================
plt.figure()
plt.imshow(A, cmap='gray')
plt.title("Original Image A")
plt.axis('off')
plt.savefig("image_A.png", bbox_inches='tight')
plt.close()

# ==============================
# (b) Singular values
# ==============================
plt.figure()
plt.semilogy(S)
plt.title("Singular Values")
plt.xlabel("Index")
plt.ylabel("Value (log scale)")
plt.savefig("singular_values.png", bbox_inches='tight')
plt.close()

# ==============================
# (c) Rank-one outer products
# ==============================
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for i in range(4):
    ui = U[:, i].reshape(-1, 1)
    vi = VT[i, :].reshape(1, -1)
    rank_one = S[i] * ui @ vi

    axes[i].imshow(rank_one, cmap='gray')
    axes[i].set_title(f"i = {i+1}")
    axes[i].axis('off')

plt.savefig("rank_one_components.png", bbox_inches='tight')
plt.close()

# ==============================
# (d) Low-rank approximations
# ==============================
k_values = [10, 20, 30, 40]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, k in enumerate(k_values):
    Ak = np.zeros_like(A)
    for i in range(k):
        ui = U[:, i].reshape(-1, 1)
        vi = VT[i, :].reshape(1, -1)
        Ak += S[i] * ui @ vi

    axes[idx].imshow(Ak, cmap='gray')
    axes[idx].set_title(f"k = {k}")
    axes[idx].axis('off')

plt.savefig("low_rank_approximations.png", bbox_inches='tight')
plt.close()

# ==============================
# (e) A, A100, log error
# ==============================
k = 100
A100 = np.zeros_like(A)

for i in range(k):
    ui = U[:, i].reshape(-1, 1)
    vi = VT[i, :].reshape(1, -1)
    A100 += S[i] * ui @ vi

error = np.abs(A - A100)
log_error = np.log(error + 1e-10)  # avoid log(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(A, cmap='gray')
axes[0].set_title("Original A")
axes[0].axis('off')

axes[1].imshow(A100, cmap='gray')
axes[1].set_title("A100")
axes[1].axis('off')

axes[2].imshow(log_error, cmap='gray')
axes[2].set_title("log |A - A100|")
axes[2].axis('off')

plt.savefig("original_vs_approx.png", bbox_inches='tight')
plt.close()

print("All figures saved successfully.")