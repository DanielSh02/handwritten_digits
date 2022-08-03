import numpy as np

# sig(x) -- returns sigmoid(x)
def sig(x):
    return 1 / (1 + np.exp(-x))

# dsig(x) -- returns sigmoid'(x)
def dsig(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))