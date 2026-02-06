import pandas as pd
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# importing dataset
skiing = pd.read_excel('skiing_dist.xlsx', index_col=0)

print(skiing.head())   # first 5 rows
print(skiing.shape)    # (rows, columns)
print(skiing.columns)  # column names

# distance matrix
Dist = skiing.to_numpy(dtype=float)
#print(Dist)
#print(Dist - Dist.T)

A = (-1/2) * (Dist * Dist )

n = Dist.shape[0]
H = np.eye(n) - (1. / n) * np.ones((n,1)) @ np.ones((n,1)).T

B = H @ A @ H

print(0.1>np.sum(B, axis=0))
print(0.1>np.sum(B, axis=1))
# if prints all true, then mean xentering worked!