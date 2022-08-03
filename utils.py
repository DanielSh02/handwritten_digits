import numpy as np
import json

# sig(x) -- returns sigmoid(x)
def sig(x):
    return 1 / (1 + np.exp(-x))

# dsig(x) -- returns sigmoid'(x)
def dsig(x):
    return np.multiply(sig(x), 1 - sig(x))


# Encodes numpy arrays for json
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)