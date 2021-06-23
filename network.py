import numpy as np
import math
from random import randint


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def msigmoid(array):
    for i in range(len(array)):
        array[i] = sigmoid(array[i])
    print(array)
    return array


class Network:
    def __init__(self):
        self.weights_matrices = [None] * 3
        self.bias_matrices = [None] * 3
        self.initialize_weights()
        self.randomize_weights()

    def initialize_weights(self):
        self.weights_matrices[0] = np.zeros((16, 784))
        self.weights_matrices[1] = np.zeros((16, 16))
        self.weights_matrices[2] = np.zeros((10, 16))

        self.bias_matrices[0] = np.zeros(16)
        self.bias_matrices[1] = np.zeros(16)
        self.bias_matrices[2] = np.zeros(10)

    def randomize_weights(self):
        for weights in self.weights_matrices:
            for row in weights:
                for i in range(len(row)):
                    row[i] = randint(1, 100) / 1000
        for biases in self.bias_matrices:
            for i in range(len(biases)):
                biases[i] = randint(1, 100) / 1000

    def evaluate(self, image) -> list[int]:
        layers = [None] * 4
        layers[0] = image
        for i in range(3):
            # print('Evaluating layer: ' + str(i))
            # print(self.weights_matrices[i])
            # print(layers[i])
            layers[i + 1] = msigmoid(np.add(np.matmul(self.weights_matrices[i], layers[i]), self.bias_matrices[i]))
        return layers[-1]

    def cost(self, image) -> int:
        cost = 0
        output = self.evaluate(image)
        correct = 100 # TODO: Match with label
        for i in range(10):
            if i == correct:
                cost += (1 - output[i]) ** 2
            else:
                cost += output[i] ** 2
        return cost

    def batch_cost(self, batch) -> int:
        total_cost = 0
        for image in batch:
            total_cost += self.cost(image)
        return total_cost