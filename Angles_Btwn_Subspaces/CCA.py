import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# -----------------------------
# Assume X, Y are already loaded as NumPy arrays
# X: m x nx
# Y: m x ny
# -----------------------------
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/cat_dogs.mat")
data = data["Y"]
data =data.astype(np.float64)/255
print(data.shape)

# Split first 99 rows as X, last 99 rows as Y
X = data[:,:99 ]
Y = data[:,-99:]

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
S_mat = S_mat.T
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

image_shape = (64, 64)  # adjust if your images are not 64x64

# Plot alpha as image
plt.figure(figsize=(6, 6))
plt.imshow(alpha.reshape(image_shape).T, cmap='gray')
plt.title("Alpha canonical variable as image")
plt.axis('off')
plt.show()

# Plot beta as image
plt.figure(figsize=(6, 6))
plt.imshow(beta.reshape(image_shape).T, cmap='gray')
plt.title("Beta canonical variable as image")
plt.axis('off')
plt.show()

# plot gamma as an image
plt.figure(figsize=(6, 6))
plt.imshow((beta * alpha).reshape(image_shape).T, cmap='gray')
plt.title("Gamma canonical variable as image")
plt.axis('off')
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