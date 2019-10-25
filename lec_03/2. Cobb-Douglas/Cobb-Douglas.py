import pandas as pd
import numpy as np
import tensorflow as tf

# Input data
DataFrame = pd.read_excel('Cobb-Douglas.xls', header=None, skiprows=1)

DataFrame = np.asarray(DataFrame[:])

n = DataFrame.shape[0]

# X
L = DataFrame[:, 2].reshape(n, -1)
K = DataFrame[:, 3].reshape(n, -1)

# y
P = DataFrame[:, 1].reshape(n, -1)

# Create random b and alpha
b = np.random.rand()
alpha = np.random.rand()

