from scipy.io import loadmat
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/cat_dogs.mat")
data = data["Y"]
data =data.astype(np.float64)/255
## 4096 X 198
print(data.shape)
plot_mode = 'weighted'
# adding labels!
n_samples = data.shape[1]  # 198
n_cats = n_samples // 2          # 99
n_dogs = n_samples - n_cats      # 99

true_labels = np.array([0]*n_cats + [1]*n_dogs)  # 0=cat, 1=dog
print(true_labels.shape)  # (198,)
print(true_labels[:10])   # first 10 labels

## To begin, we need a distance matrix. I am pulling one from the MDS jupyter ntoebook on canvas
def distmat(X, method='2norm'):
    n = X.shape[1]
    D = np.zeros((n, n))

    if method == '2norm':
        for i in range(n - 1):
            for j in range(i + 1, n):
                D[i, j] = np.linalg.norm(X[:, i] - X[:, j])
    elif method == 'angle':
        for i in range(n - 1):
            for j in range(i + 1, n):
                x = X[:, i]
                y = X[:, j]
                D[i, j] = np.arccos(np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
                # Alternatively, use D(i,j) = min(j-i, n+i-j) * 2 * np.pi / 7
    else:
        print('Unknown distance method')

    D += D.T
    return D


Dist = distmat(data, method='2norm')
Dist = Dist.astype(np.float64)/255

n = Dist.shape[0]

k = 100
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

# Plotting (thanks CHATGPT)
plt.figure(figsize=(8,6))

cats = true_labels == 0
dogs = true_labels == 1
if plot_mode == 'unweighted' or plot_mode == 'both':
    plt.scatter(embedding_A[cats,0], embedding_A[cats,1], marker='o', label='Cats (Unweighted)')
    plt.scatter(embedding_A[dogs,0], embedding_A[dogs,1], marker='x', label='Dogs (Unweighted)')

if plot_mode == 'weighted' or plot_mode == 'both':
    plt.scatter(embedding_W[cats,0], embedding_W[cats,1], marker='o', label='Cats (Weighted)')
    plt.scatter(embedding_W[dogs,0], embedding_W[dogs,1], marker='x', label='Dogs (Weighted)')

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("2D Laplacian Eigenmaps")
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8,6))

cats = true_labels == 0
dogs = true_labels == 1

if plot_mode == 'unweighted' or plot_mode == 'both':
    # Using subset lengths (0…n_cats-1, 0…n_dogs-1)
    plt.scatter(range(embedding_A[cats].shape[0]), embedding_A[cats,0], marker='o', label='Cats (Unweighted)')
    plt.scatter(range(embedding_A[dogs].shape[0]), embedding_A[dogs,0], marker='x', label='Dogs (Unweighted)')
if plot_mode == 'weighted' or plot_mode == 'both':
    # Cats
    plt.scatter(range(embedding_W[cats].shape[0]), embedding_W[cats, 0], marker='o', label='Cats (Weighted)')
    # Dogs
    plt.scatter(range(embedding_W[dogs].shape[0]), embedding_W[dogs, 0], marker='x', label='Dogs (Weighted)')
plt.xlabel("Image Index")
plt.ylabel("Fiedler Embedding")
plt.title("Fiedler Vector Embedding")
plt.legend()
plt.grid(True)
plt.show()

# Fiedler vector
fiedler = eigvecs_W[:,1]
print('Max fiedler is')
print(np.max(fiedler))
print('Min. fiedler is')
print(np.min(fiedler))

threshold_range = np.linspace(np.min(fiedler), np.max(fiedler), 100)
accuracy_log = np.zeros(len(threshold_range))
i=0 # counter
for threshold in threshold_range:
    fiedler_labels = (fiedler > threshold).astype(int)
    #print(fiedler_labels)
    # Compute accuracy accounting for possible flip
    acc1 = np.sum(fiedler_labels == true_labels) / len(true_labels)
    acc2 = np.sum(1 - fiedler_labels == true_labels) / len(true_labels)  # flipped
    accuracy = max(acc1, acc2)
    accuracy_log[i] = accuracy
    i += 1
print("Fiedler cut accuracy (% of correct labels):", np.max(accuracy_log) * 100)
print(np.where(accuracy_log == np.max(accuracy_log)))
print(np.linspace(np.min(fiedler), np.max(fiedler), 100)[np.where(accuracy_log == np.max(accuracy_log))])



# this function creates a figure with a few random cat images. From C(h)atGPT

def plot_sample_images_grid(X, labels=None, n_rows=3, n_cols=5, img_shape=(64, 64), cmap='gray'):
    """
    Plots a grid of sample images from the dataset.

    Parameters:
    - X: np.array of shape (4096, n_samples_total)
    - labels: optional, array of 0/1 labels (0=cat, 1=dog)
    - n_rows: number of rows in the grid
    - n_cols: number of columns in the grid
    - img_shape: tuple, shape of each image (default 64x64)
    - cmap: color map (default gray)
    """
    n_total = X.shape[1]
    n_samples = n_rows * n_cols
    indices = np.random.choice(n_total, size=n_samples, replace=False)

    plt.figure(figsize=(2 * n_cols, 2 * n_rows))

    for i, idx in enumerate(indices):
        img = X[:, idx].reshape(img_shape)
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        if labels is not None:
            plt.title('Cat' if labels[idx] == 0 else 'Dog', fontsize=8)

    plt.tight_layout()
    plt.show()
# ---------------- Example Usage ----------------
plot_sample_images_grid(data, labels=true_labels, n_rows=3, n_cols=5)