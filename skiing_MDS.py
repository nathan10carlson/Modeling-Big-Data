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
print(Dist)
print(Dist - Dist.T)