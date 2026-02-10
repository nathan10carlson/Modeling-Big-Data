import numpy as np

# Incidence matrix Q
Q = np.array([
    [ 1,  0,  1,  0,  0, -1],
    [ 0,  1,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0],
    [ 0, -1,  1,  1,  0,  0],
    [ 0,  0, -1,  0,  1,  0],
    [ 0,  0,  0, -1, -1,  0],
    [ 0,  0,  0,  0,  0,  1]
])

# Compute the rank
rank_Q = np.linalg.matrix_rank(Q)
print("Rank of Q:", rank_Q)

# Number of nodes
n = Q.shape[0]

# Number of connected components
num_components = n - rank_Q
print("Number of connected components:", num_components)