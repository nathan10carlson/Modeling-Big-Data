import numpy as np
import matplotlib.pyplot as plt
import pywt

# First shape (edges only)
A1 = np.zeros((8, 8))
A1[2, 2:6] = 1
A1[5, 2:6] = 1
A1[2:6, 2] = 1
A1[2:6, 5] = 1
print(A1)

### option
A1_checkerboard = False


if A1_checkerboard:
    A1 = np.zeros((8, 8))

    # filled square (low frequency region)
    A1[1:5, 1:5] = 1

    # checkerboard pattern (high frequency)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                A1[i, j] += 0.5

    # vertical stripe
    A1[:, 6] = 1

    # horizontal stripe
    A1[6, :] = 1

# Second shape (filled square)
A2 = np.zeros((8, 8))
A2[2:6, 2:6] = 1

# Plot
plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(A1)
plt.title("Edges Only")

plt.subplot(1, 2, 2)
plt.imshow(A2)
plt.title("Filled Square")

plt.show()



## Making a function that gets high and low pass
def wavelet(x):
    len = x.shape[0]

    def haar_block_matrix(n):
        assert n % 2 == 0, "n must be even"

        H = np.zeros((n, n))

        for i in range(0, n, 2):
            H[i, i] = 1
            H[i, i + 1] = 1
            H[i + 1, i] = 1
            H[i + 1, i + 1] = -1

        return H / np.sqrt(2)

    H = haar_block_matrix(len)
    #print(H)

    U_L1 = np.zeros_like(H[:,H.shape[0]//2:])
    for i in range(0, H.shape[0]//2):
        U_L1[:, i] = H[:,2*i]
    #print(U_L1)
    U_H1 = np.zeros_like(H[:,H.shape[0]//2:])
    for i in range(0, H.shape[0] // 2):
        U_H1[:, i] = H[:,1 + 2 * i]
    #print(U_H1)
    return U_L1, U_H1
U_L1, U_H1 = wavelet(A1[:,4])
orig = A1
low = U_L1@U_L1.T @ A1
high = U_H1@U_H1.T @ A1

## combine to visualize separation

# Plot
plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(A1)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(high)
plt.title("U_H1 Reconstruction (Columns)")

plt.subplot(1, 3, 3)
plt.imshow(low)
plt.title("U_L1 Reconstruction (Columns)")
plt.show()

LL = U_L1 @ U_L1.T @ A1 @ U_L1 @ U_L1.T
LH = U_L1 @ U_L1.T @ A1 @ U_H1 @ U_H1.T
HL = U_H1 @ U_H1.T @ A1 @ U_L1 @ U_L1.T
HH = U_H1 @ U_H1.T @ A1 @ U_H1 @ U_H1.T
print('divide')
recon = LL + LH + HL + HH

print(np.allclose(A1, recon))

fig, axes = plt.subplots(2, 3, figsize=(10, 6))

def show(ax, img, title):
    vmax = np.max(np.abs(img)) + 1e-8  # avoid divide-by-zero
    im = ax.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

show(axes[0, 0], A1, "Original")
show(axes[0, 1], LL, "LL (Low-Low)")
show(axes[0, 2], LH, "LH (Low-High)")
show(axes[1, 0], HL, "HL (High-Low)")
show(axes[1, 1], HH, "HH (High-High)")

axes[1, 2].axis("off")

plt.tight_layout()
plt.show()


LL = U_L1 @ U_L1.T @ A2 @ U_L1 @ U_L1.T
LH = U_L1 @ U_L1.T @ A2 @ U_H1 @ U_H1.T
HL = U_H1 @ U_H1.T @ A2 @ U_L1 @ U_L1.T
HH = U_H1 @ U_H1.T @ A2 @ U_H1 @ U_H1.T


fig, axes = plt.subplots(2, 3, figsize=(10, 6))

def show(ax, img, title):
    vmax = np.max(np.abs(img)) + 1e-8  # avoid divide-by-zero
    im = ax.imshow(img, cmap='seismic', vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

show(axes[0, 0], A2, "Original")
show(axes[0, 1], LL, "LL (Low-Low)")
show(axes[0, 2], LH, "LH (Low-High)")
show(axes[1, 0], HL, "HL (High-Low)")
show(axes[1, 1], HH, "HH (High-High)")

axes[1, 2].axis("off")

plt.tight_layout()
plt.show()