import GSVD_goated_script as GSVD

import numpy as np

A = np.array([[1,2],
              [0, 1],
              [2, 1],
              [0, -1],
              [1, 1]])

B = np.array([[1, 1],
              [0, 1],
              [2, 0],
              [0, 1]]
)

U, V, C, S, X = GSVD.GSVD(A, B)

print(C.T@C)
print(S.T@S)
