import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
A = np.random.rand(33,3)
B = np.random.rand(34,3)
rank_A = np.linalg.matrix_rank(A)
rank_B = np.linalg.matrix_rank(B)


def GSVD(A, B, thin = True, print_dims = False):
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    m = A.shape[0]
    n = A.shape[1]
    p = B.shape[0]
    print('m=',m)
    print('n=',n)
    print('p=',p)

    M = np.concatenate((A, B), axis=0)
    k = np.linalg.matrix_rank(M)
    l = m+p # num total rows
    M_decomp = np.linalg.svd(M, full_matrices=thin)

    Q_hat = M_decomp[0] # U
    Sigma_hat = np.diag(M_decomp[1]) # Sig
    Sigma_k = Sigma_hat[:k,:k]
    W_T = M_decomp[2] # V_T
    W = W_T.T

    W_1 = W[:, :k]
    W_2 = W[:, k:]
    print("Shapes:")
    print("Q_hat shape:", Q_hat.shape)
    print("Sigma_hat shape:", Sigma_hat.shape)
    print("W_T shape:", W_T.shape)

    Q_1_1 = Q_hat[0:m,:k]
    Q_2_1 = Q_hat[m:l, :k]

    U, V, C, S, X = CSAlgorithm(Q_1_1, Q_2_1)
    G = np.hstack((W_1 @ Sigma_k @ X, W_2))
    if print_dims:
        print('m:', m)

    return U, V, C, S, G

def CSAlgorithm(Q_1_1, Q_2_1):
    Flag = 0  # Default is Thin

    m, k = Q_1_1.shape
    print('m=',m)
    print('k=',k)
    p, _ = Q_2_1.shape
    print('p=',p)

    if Flag == 0:  # thin
        U, C, Xt = np.linalg.svd(Q_1_1, full_matrices=False)
        C = np.diag(C)
        X = Xt.T
        c_vals = np.diag(C)
        tol = 1e-10
        r = np.sum(c_vals < tol)
        print('r=',r)

        S = np.zeros((k, k))
        F = np.random.randn(p, r)

    else:  # full
        U, C, Xt = np.linalg.svd(Q_1_1, full_matrices=True)
        C = np.diag(C)
        X = Xt.T
        S = np.zeros((p, k))
        r = k - np.linalg.matrix_rank(np.diag(C))
        print('r=',r)
        F = np.random.randn(p, p - k + r)

    S_hat = np.zeros((k, k))


    Vp_list = []

    for i in range(k):
        S_hat[i, i] = np.sqrt(1 - C[i,i]**2)

        if S_hat[i, i] > 0:
            Y = Q_2_1 @ X
            v = Y[:, i] / S_hat[i, i]
            Vp_list.append(v)

    if len(Vp_list) > 0:
        Vp = np.column_stack(Vp_list)
    else:
        Vp = np.zeros((p, 0))

    r = k - Vp.shape[1]
    print('r=',r)

    if Flag == 0:
        F = np.random.randn(p, r)
    else:
        F = np.random.randn(p, p - Vp.shape[1])

    # Orthogonal complement construction
    Vt = F - Vp @ (Vp.T @ F)
    Vperp, _ = np.linalg.qr(Vt, mode='reduced')

    V = np.column_stack((Vperp, Vp))

    if Flag == 0:
        S = S_hat
    else:
        OO = np.zeros((p - k, k))
        S = np.vstack((OO, S_hat))

    c_hat = np.diag(C)
    s_hat = np.diag(S)
    s = (np.sum((1>np.diag(S)) & (np.diag(S)>0)))
    print('s=',s)

    print('s:',s)

    print('pairs:')
    for c, s in zip(c_hat, s_hat):
        print(c, s)
        check = c**2 + s**2
        print(check)

    return U, V, C, S, X


U, V, C, S, G = GSVD(A, B)


print("U orthogonality:", np.linalg.norm(U.T @ U - np.eye(U.shape[1])))
print("V orthogonality:", np.linalg.norm(V.T @ V - np.eye(V.shape[1])))

G_inv = np.linalg.pinv(G)

print("A reconstruction error:", np.linalg.norm(A - U @ C @ G.T))
print("B reconstruction error:", np.linalg.norm(B - V @ S @ G.T))
print("CS identity error:", np.linalg.norm(C**2 + S**2 - np.eye(C.shape[0])))