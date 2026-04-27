from toy_data import gen_toy_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linprog
import numpy as np


# -----------------------------
# generate data
# -----------------------------
plot_data_option = False
toy_data_plot = False

toy_data = gen_toy_data(40, 40, plot=plot_data_option)

print(toy_data.shape)

y = toy_data[:, -1]
y_labels = y.reshape(-1, 1)

print('y_labels = ', y_labels.shape)

y_diag = np.diag(y)
print(y_diag)

C = 1

X = toy_data[:, :-1]
print(X)


# =========================================================
# Sparse SVM FUNCTION (clean structure only)
# =========================================================
def Sparse_SVM(data, C=1, plot_data_option=False):

    # -----------------------------
    # split data
    # -----------------------------
    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    n = X.shape[1]
    m = y.shape[0]

    I_n = np.eye(n)
    neg_I_n = -1 * I_n
    I_m = np.eye(m)

    O_nm = np.zeros((n, m))
    O_mn = np.zeros((m, n))
    O_n_1 = np.zeros((n, 1))

    Y_diag = np.diag(y.flatten())
    YX = Y_diag @ X

    # -----------------------------
    # build constraint matrix
    # -----------------------------
    row_1 = np.concatenate((I_n, neg_I_n, O_nm, O_n_1), axis=1)
    row_2 = np.concatenate((neg_I_n, neg_I_n, O_nm, O_n_1), axis=1)
    row_3 = -1 * np.concatenate((YX, O_mn, I_m, y), axis=1)

    A_full = np.concatenate((row_1, row_2, row_3), axis=0)

    # -----------------------------
    # RHS vector
    # -----------------------------
    h = np.zeros((A_full.shape[0], 1))
    h[2*n:, 0] = -1

    # -----------------------------
    # objective
    # -----------------------------
    c = np.concatenate(
        (np.zeros((n, 1)),
         np.ones((n, 1)),
         C * np.ones((m, 1)),
         np.zeros((1, 1))),
        axis=0
    )

    # -----------------------------
    # bounds
    # -----------------------------
    unbounded = (None, None)
    bounded = (0, None)

    bounds = (
        [unbounded] * n +
        [bounded] * (n + m) +
        [unbounded]
    )

    # -----------------------------
    # solve LP
    # -----------------------------
    solution = linprog(
        c.flatten(),
        A_ub=A_full,
        b_ub=h.flatten(),
        bounds=bounds,
        method="highs"
    )

    x = solution.x

    weights = x[0:n]
    bias = x[-1]

    print("Weights", weights)
    print("Bias", bias)

    # -----------------------------
    # optional visualization
    # -----------------------------
    if plot_data_option:
        A_aug = np.hstack((A_full, h))

        plt.figure(figsize=(12, 6))
        sns.heatmap(A_aug, cmap="coolwarm", center=0)

        plt.title("Sparse SVM: A_full with RHS vector h")
        plt.xlabel("Variables + RHS")
        plt.ylabel("Constraints")
        plt.show()

    return weights, bias


# =========================================================
# RUN MODEL
# =========================================================
if toy_data_plot:
    weights, bias = Sparse_SVM(toy_data, C=1, plot_data_option=False)


    # =========================================================
    # PLOTTING DECISION BOUNDARY (cleaned only formatting)
    # =========================================================
    X_plot = toy_data[:, :-1]
    y_plot = toy_data[:, -1]

    w1, w2 = weights[0], weights[1]

    plt.figure(figsize=(7, 6))

    plt.scatter(X_plot[y_plot == 1, 0], X_plot[y_plot == 1, 1], label="+1 class")
    plt.scatter(X_plot[y_plot == -1, 0], X_plot[y_plot == -1, 1], label="-1 class")

    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
    xx = np.linspace(x_min, x_max, 200)

    if abs(w2) > 1e-8:
        yy = -(w1 * xx + bias) / w2
        plt.plot(xx, yy, 'k', linewidth=2, label="decision boundary")
    else:
        x0 = -bias / w1
        plt.axvline(x0, color='k', label="decision boundary")

    plt.title("Sparse SVM Decision Boundary (LP solution)")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.show()