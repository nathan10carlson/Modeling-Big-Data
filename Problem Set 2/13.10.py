import numpy as np
import matplotlib.pyplot as plt
import GSVD_goated_script as GSVD
from scipy.io import loadmat


data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/indian_pines_gt.mat")

cube = data["indian_pines_gt"]
print(cube.shape)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load ground truth
data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/indian_pines_gt.mat")
gt = data["indian_pines_gt"]

# Plot image
plt.figure(figsize=(6, 6))
plt.imshow(gt, cmap='jet')
plt.title("Indian Pines Ground Truth")
plt.colorbar()
plt.axis('off')

plt.show()

data = loadmat("/Users/nathancarlson/Desktop/programs/MATH 532/data/indian_pines_corrected.mat")
print(data.keys())
to_analyze = data["indian_pines_corrected"]
print(to_analyze.shape)
a_25 = to_analyze[:,:,24]
# Plot image
plt.figure(figsize=(6, 6))
plt.imshow(a_25, cmap='jet')
plt.title(r"Indian Pines ({a_{25}}")
plt.colorbar()
plt.axis('off')
plt.show()

# extract band a25

# compute finite differences
dx = np.zeros_like(a_25)
# horizontal differences
dx[:, :-1] += a_25[:, 1:] - a_25[:, :-1]
# vertical differences
dx[:-1, :] += a_25[1:, :] - a_25[:-1, :]

# normalize
N = dx / np.sqrt(2)
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(N, cmap='gray')
plt.title("Estimated Noise N = (1/sqrt(2)) dX")
plt.axis('off')
plt.colorbar()
plt.show()

U, V, C, S, G = GSVD.GSVD(a_25, N)
print(U[:,])
print("cshape", C.shape)
import numpy as np
import matplotlib.pyplot as plt

c = np.diag(C)
s = np.diag(S)
i = np.arange(len(c))

plt.figure(figsize=(8,5))
plt.plot(i, c, marker='o', label='c_i (signal)')
plt.plot(i, s, marker='o', label='s_i (noise)')

plt.xlabel("Index i")
plt.ylabel("Value")
plt.title("GSVD Singular Value Pairs vs Index")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(i, c, marker='o', label='c_i (signal)')
plt.plot(i, s, marker='o', label='s_i (noise)')

plt.xlabel("Index i")
plt.xlim(0,5)
plt.ylabel("Value")
plt.title("GSVD Singular Value Pairs vs Index")
plt.legend()
plt.grid(True)
plt.show()

print('Looks like we should use 2')
print(U[:,2].shape)

i = 2
rank_i_approx = U[:,:i] @ C[:i,:i] @ G[:,:i].T

plt.figure(figsize=(6, 6))
plt.imshow(rank_i_approx, cmap='gray')
plt.title(rf"Rank {i} Approximation")
plt.axis('off')
plt.colorbar()
plt.show()

c = np.diag(C)
s = np.diag(S)


