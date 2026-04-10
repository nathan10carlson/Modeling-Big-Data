from scipy.io import loadmat
import numpy as np
from scipy.linalg import qr
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/cat_dogs.mat")
cats = data["Y"]
cats = cats.astype(np.float64) / 255.0
# 4096 x 198

# ----------------------------
# Random sampling
# ----------------------------
idx = np.random.choice(99, size=21, replace=False)
rand_dog_idx = np.random.choice(99) + 98  # dog index

# ----------------------------
# Build subspaces
# ----------------------------
cat_1 = cats[:, idx[:6]]
cat_2 = cats[:, idx[6:13]]
cat_3 = cats[:, idx[13:]]

dog_to_add = cats[:, rand_dog_idx].reshape(-1, 1)

cat_1 = np.concatenate((cat_1, dog_to_add), axis=1)
cat_2 = np.concatenate((cat_2, dog_to_add), axis=1)
cat_3 = np.concatenate((cat_3, dog_to_add), axis=1)

print("cat_1 shape:", cat_1.shape)
print("cat_2 shape:", cat_2.shape)
print("cat_3 shape:", cat_3.shape)

# ----------------------------
# Orthonormalization
# ----------------------------
def orthonormalize(X):
    Q, _ = qr(X, mode='economic')
    return Q

cat_1 = orthonormalize(cat_1)
cat_2 = orthonormalize(cat_2)
cat_3 = orthonormalize(cat_3)

# ----------------------------
# Plot + save image grids
# ----------------------------
def plot_image_grid(X, title, filename, cols=3):
    n = X.shape[1]
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        axes[i].axis('off')
        if i < n:
            img = X[:, i].reshape(64, 64)
            axes[i].imshow(img, cmap='gray')

    fig.suptitle(title)
    plt.tight_layout()

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_image_grid(cat_1, "Cat Group 1 (with dog)", "cat_group_1.png")
plot_image_grid(cat_2, "Cat Group 2 (with dog)", "cat_group_2.png")
plot_image_grid(cat_3, "Cat Group 3 (with dog)", "cat_group_3.png")

# ----------------------------
# Concatenate all subspaces
# ----------------------------
X = np.concatenate((cat_1, cat_2, cat_3), axis=1)

# ----------------------------
# SVD / Flag mean
# ----------------------------
U, S, Vt = np.linalg.svd(X, full_matrices=False)

k = 3
Mk = U[:, :k]

flag_means = [Mk[:, i] for i in range(k)]

# ----------------------------
# Plot + save flag means
# ----------------------------
fig, axes = plt.subplots(1, k, figsize=(12, 4))

for i in range(k):
    img = flag_means[i].reshape(64, 64)

    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Flag mean {i+1}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("flag_means.png", dpi=300, bbox_inches='tight')
plt.show()