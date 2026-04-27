from sparse_SVM_custom import Sparse_SVM
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

plot_pca = False
create_test_data = False
data_setname = 'b-cancer_data'
C = 1

file_path = "/Users/nathancarlson/Desktop/programs/MATH 532/sparse_SVM/wisconsin_breast_cancer.mat"

if create_test_data:
    data_mat = loadmat(file_path)

    #print(data_mat.keys())

    #print(data_mat['feature_names'])
    feature_names = data_mat['feature_names']
    feature_names = np.array([f[0] for f in feature_names.flatten()])


    samples = data_mat['X']
    labels = data_mat['y']
    labels = labels.reshape(-1, 1)
    # making all 0 labels 1
    labels = np.where(labels == 0, -1, 1)
    data = np.concatenate((samples, labels), axis=1)

    from sklearn.model_selection import train_test_split
    import numpy as np

    # -----------------------------
    # Data
    # -----------------------------
    X = samples
    y = labels.ravel()

    # -----------------------------
    # Train / Temp split (80 / 20)
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Split temp into val / test (50 / 50)
    # -> 10% val, 10% test total
    # -----------------------------
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # -----------------------------
    # Save everything
    # -----------------------------
    np.savez(
        data_setname,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feature_names
    )

    print(f"Saved dataset to {data_setname}.npz")

data = np.load(f"{data_setname}.npz")

X_train = data["X_train"]
y_train = data["y_train"]

X_val = data["X_val"]
y_val = data["y_val"]

X_test = data["X_test"]
y_test = data["y_test"]

feature_names = data["feature_names"]

train_data = np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1)

weights, bias = Sparse_SVM(train_data, C=C, plot_data_option=True)

# -----------------------------
# Prediction function
# -----------------------------
def predict(X, w, b):
    scores = X @ w + b
    preds = np.sign(scores)
    preds[preds == 0] = 1  # or -1, but be consistent
    return preds

# -----------------------------
# Compute accuracies
# -----------------------------
y_train_pred = predict(X_train, weights, bias)
y_val_pred   = predict(X_val, weights, bias)

train_acc = np.mean(y_train_pred == y_train)
val_acc   = np.mean(y_val_pred == y_val)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Pair weights with features

weights = np.array(weights).flatten()
abs_weights = np.abs(weights)

## order features from greatest weight to least
# sort indices (largest → smallest)
idx = np.argsort(abs_weights)[::-1]
ordered_features = feature_names[idx]
ordered_abs_weights = abs_weights[idx]

k = 10

ordered_weights = weights[idx]
print(r'Printing Top k features')

for f, w in zip(ordered_features[:k], ordered_weights[:k]):
    print(f, w)

## Plotting magintude of Features
plt.figure()
plt.xlabel(r"$k$ most important feature (greatest to least)")
plt.ylabel("Absolute Value of Feature Weights (greatest to least)")
plt.plot(np.arange(1,len(ordered_abs_weights)+1, 1),ordered_abs_weights +1e-12)
plt.title("Absolute Value of Feature Weights")
#plt.yscale('log')
plt.show()

if plot_pca:
    # -----------------------------
    # 1. Standardize features (VERY important for PCA)
    # -----------------------------
    X = samples
    y = labels.ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # 2. Fit PCA
    # -----------------------------
    pca = PCA(n_components=2)  # 2D projection
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Total variance captured:", np.sum(pca.explained_variance_ratio_))

    # -----------------------------
    # 3. Plot PCA projection
    # -----------------------------
    plt.figure()

    plt.scatter(X_pca[y == -1, 0], X_pca[y == -1, 1], label="Class -1", alpha=0.5)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], label="Class +1", alpha=0.5)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (2D)")
    plt.legend()
    plt.show()

