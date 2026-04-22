import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# -----------------------------
# Assume X, Y are already loaded as NumPy arrays
# X: m x nx
# Y: m x ny
# -----------------------------

## Taking samples as rows

corn_data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/IPcornnotilC2.mat")
corn_data = corn_data["X"].T


woods_data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/IPwoodsC14.mat")
woods_data = woods_data["Y"].T
print(woods_data.shape)
# Split first 99 rows as X, last 99 rows as Y

# calling corn X, wood is Y, taking first `265
length = min(woods_data.shape[0], corn_data.shape[0])
X = corn_data[:length,:]
Y = woods_data[:length,:]

print("X shape:", X.shape)
print("Y shape:", Y.shape)


m = X.shape[0]
nx = X.shape[1]
ny = Y.shape[1]

# Subtract row means (center each variable)
XX = (np.eye(m) - np.ones((m, m)) / m) @ X
YY = (np.eye(m) - np.ones((m, m)) / m) @ Y

# QR decomposition
Qx, Rx = np.linalg.qr(XX, mode='reduced')
Qy, Ry = np.linalg.qr(YY, mode='reduced')

# SVD of cross-correlation of Q's
R_mat, D, S_mat = np.linalg.svd(Qx.T @ Qy)

# First correlation coefficient
z = D[0] # theta_max
print("Correlation coefficient:", z)

# Angle in radians
angle = np.arccos(D[0])
print("Angle (radians):", angle)

# Canonical vectors
a = np.linalg.solve(Rx, R_mat[:, 0])
b = np.linalg.solve(Ry, S_mat[:, 0])

# Correlation variables
alpha = XX @ a
beta = YY @ b
print(alpha.shape, beta)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6, 6))
plt.plot(alpha, beta, '.', label='Canonical variables')
x_line = np.arange(-0.05, 0.051, 0.01)  # may adjust depending on scale
plt.plot(x_line, x_line, 'k', label='y = x')
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('Canonical Correlation Plot')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(6, 6))
plt.plot(np.arange(len(a)), a, label='a components')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Canonical Vector a Components')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(6, 6))
plt.plot(np.arange(len(b)), b, label='b components')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Canonical Vector b Components')
plt.grid(True)
plt.legend()
plt.show()