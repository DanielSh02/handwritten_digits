import network
import numpy as np
from random import randint

test_image = np.zeros(784)
for i in range(784):
    test_image[i] = randint(0, 1)

network = network.Network()

print(network.evaluate_nodes(test_image))

