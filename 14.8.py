import numpy as np

# Number of nodes

Q_manual = np.array([
    [ 1,  1,  1,  0,  0, -1,  0,  0,  0,  0],
    [-1,  0,  0,  1,  1,  0,  0,  0, -1,  0],
    [ 0,  0,  0,  0, -1,  1,  1,  0,  0,  0],
    [ 0, -1,  0, -1,  0,  0,  0,  1,  0, -1],
    [ 0,  0,  0,  0,  0,  0, -1, -1,  1,  1],
    [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0]
], dtype=int)
print("\nIncidence matrix Q_manual:\n")
QQ_T = Q_manual @ Q_manual.T
print(QQ_T)

# computing D
Q_abs