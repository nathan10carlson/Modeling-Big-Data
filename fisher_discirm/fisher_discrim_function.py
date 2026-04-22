import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# Function to generate sample data
# -----------------------------
def generate_data(n_samples_1=3, n_samples_2=3, dim=2):
    class_1 = np.random.randint(0, 10, size=(n_samples_1, dim))
    class_2 = np.random.randint(5, 30, size=(n_samples_2, dim))
    labels_1 = np.zeros(n_samples_1)
    labels_2 = np.ones(n_samples_2)
    return class_1, class_2, labels_1, labels_2

# -----------------------------
# User input for sample size and dimension
# -----------------------------
n_samples_1 = int(input("Enter number of samples for Class 1: "))
n_samples_2 = int(input("Enter number of samples for Class 2: "))
dim = int(input("Enter dimension of data (2 or 3 recommended): "))

class_1, class_2, labels_1, labels_2 = generate_data(n_samples_1, n_samples_2, dim)

# -----------------------------
# Determine class means
# -----------------------------
class_1_mean = np.mean(class_1, axis=0)
class_2_mean = np.mean(class_2, axis=0)

# -----------------------------
# Compute within-class scatter
# -----------------------------
class_1_diff = class_1 - class_1_mean
class_2_diff = class_2 - class_2_mean
class_1_scatter = class_1_diff.T @ class_1_diff
class_2_scatter = class_2_diff.T @ class_2_diff
Sw = class_1_scatter + class_2_scatter

# -----------------------------
# Compute Fisher direction
# -----------------------------
w = np.linalg.inv(Sw) @ (class_2_mean - class_1_mean)
w = w / np.linalg.norm(w)
print("Fisher direction w:", w)

# -----------------------------
# Project data onto Fisher line
# -----------------------------
X = np.vstack((class_1, class_2))
y = np.hstack((labels_1, labels_2))
X_proj = X @ w

# -----------------------------
# Plot original data and hyperplane/line
# -----------------------------
if dim == 2:
    plt.figure()
    plt.scatter(class_1[:,0], class_1[:,1], label='Class 1')
    plt.scatter(class_2[:,0], class_2[:,1], label='Class 2')
    t_vals = np.linspace(np.min(X_proj), np.max(X_proj), 100)
    line_points = np.outer(t_vals, w)
    plt.plot(line_points[:,0], line_points[:,1], 'k--', label='Fisher hyperplane line')
    plt.title('Original Data with Fisher Line')
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

elif dim == 3:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(class_1[:,0], class_1[:,1], class_1[:,2], label='Class 1')
    ax.scatter(class_2[:,0], class_2[:,1], class_2[:,2], label='Class 2')
    t_vals = np.linspace(np.min(X_proj), np.max(X_proj), 100)
    line_points = np.outer(t_vals, w)
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'k--', label='Fisher hyperplane line')
    ax.set_title('3D Data with Fisher Line')
    ax.legend()
    plt.show()

# -----------------------------
# Plot 1D projection
# -----------------------------
plt.figure()
plt.hist(X_proj[y==0], bins=20, alpha=0.6, label='Class 1')
plt.hist(X_proj[y==1], bins=20, alpha=0.6, label='Class 2')
plt.title('Projection onto Fisher Discriminant')
plt.xlabel('Projected Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# -----------------------------
# Threshold classification based on labels
# -----------------------------
threshold = (X_proj[y==0].mean() + X_proj[y==1].mean()) / 2
preds = (X_proj > threshold).astype(int)
accuracy = np.mean(preds == y)
print(f"Threshold: {threshold:.3f}, Accuracy: {accuracy:.3f}")
