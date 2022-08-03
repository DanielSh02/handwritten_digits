from math import exp
from random import uniform, sample, shuffle
import numpy as np
from numpy import array, add, e, matmul, argmax, empty, copy, zeros
from data import *
from utils import *
from json import dump, loads

class Network:
    def __init__(self, hidden_layers, labels, raw, learning_rate = 0.1, batch_size = 20, print_statements = True):
        # Data Set
        self.data_set = DataSet(raw)

        # Layers
        self.layers = [Layer(self.data_set.input_len)]
        for i in range(len(hidden_layers)):
            self.layers.append(HiddenLayer(hidden_layers[i], self.layers[-1]))
        self.layers.append(OutputLayer(self.layers[-1], labels))

        # Weights and biases
        self.weights = [[]] + [layer.weights for layer in self.layers[1:]]
        self.biases = [[]] + [layer.biases for layer in self.layers[1:]]
        self.zs = [[]] + [layer.zs for layer in self.layers[1:]]
        self.w_batch = []
        self.b_batch = []
        self.counter = 1

        # Batches
        self.batch_cost = 0
        self.batch = []

        # Training configuration
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.learning_rate = self.learning_rate / self.batch_size
        self.print_statements = print_statements

    def run(self, data):
        self.layers[0].nodes = array([data]).transpose()
        for layer in self.layers:
            layer.eval()
        return self.layers[-1].result, self.layers[-1].confidence

    def train(self, epochs):
        dataset_size = len(self.data_set)
        for i in range(epochs):
            self.data_set.idx = 0
            self.data_set.shuffle()
            while True:
                data, actual = next(self.data_set)
                self.run(data)
                self.calc_change(actual)
                if not self.data_set.idx % self.batch_size:
                    self.backprop()
                    if self.print_statements:
                        print(f'Epoch: {i}')
                        print(actual, self.layers[-1].result, self.layers[-1].nodes)
                if self.data_set.idx >= dataset_size:
                    break

    def calc_change(self, actual):
        n = len(self.layers)
        desired = array([[1 if i == actual else 0 for i in self.labels]]).transpose()
        weight_changes = [zeros(array(w).shape) for w in self.weights]
        bias_changes = [zeros(array(b).shape) for b in self.biases]
        dCdz = np.multiply(dsig(self.zs[-1]), self.layers[-1].nodes - desired)
        bias_changes[-1] = dCdz
        weight_changes[-1] = self.layers[-2].nodes.transpose()
        for i in range(n - 2, 0, -1):
            z = self.zs[i]
            dCdz = np.multiply(dsig(z), self.weights[i + 1].transpose() @ dCdz)
            weight_changes[i] = dCdz @ self.layers[i].nodes.transpose()
            bias_changes[i] = dCdz

        self.w_batch.append(weight_changes)
        self.b_batch.append(bias_changes)
        for i in range(len(self.labels)):
            self.batch_cost += (self.layers[-1].nodes[i] - desired[i]) ** 2

    def backprop(self):
        for i, layer in enumerate(self.layers):
            if i != 0:
                for j in range(len(self.w_batch)):
                    layer.weights = add(layer.weights, -self.learning_rate * array(self.w_batch[j][i]))
                    layer.biases = add(layer.biases, -self.learning_rate * array(self.b_batch[j][i]))
        self.w_batch = []
        self.b_batch = []
        if self.print_statements:
            print(f"Batch {self.counter // self.batch_size} average cost: {self.batch_cost / self.batch_size}")
        self.batch_cost = 0

    def unload(self, file):
        data = {}
        data["weights"] = self.weights
        data["biases"] = self.biases
        with open(file, "w"):
            dump(data, file, indent = 6)

    def load(self, file):
        data = loads(file)
        self.weights = data["weights"]
        self.biases = data["biases"]
        hidden_layers = [len(arr) for arr in self.biases[:-1]]
        print(f"Loading file: {file}. Hidden Layers = {hidden_layers}")

        self.layers = [Layer(self.data_set.input_len)]
        for i in range(len(hidden_layers)):
            self.layers.append(HiddenLayer(self.hidden_layers[i], self.layers[-1]))
        self.layers.append(OutputLayer(self.layers[-1], self.labels))
        self.zs = [[]] + [layer.zs for layer in self.layers[1:]]



class Layer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = array([0 for i in range(num_nodes)]).tranpose()
        self.zs = array([0 for i in range(num_nodes)]).transpose()

class HiddenLayer(Layer):
    def __init__(self, num_nodes, prev_layer):
        super().__init__(num_nodes)
        def randizer():
            return np.random.normal(scale = self.prev_layer.num_nodes ** (-0.5))
        self.weights = array([[randizer() for i in range(prev_layer.num_nodes)] for j in range(num_nodes)])
        self.biases = array([[0 for node in range(num_nodes)]]).transpose()

    def eval(self):
        self.zs = add(self.weights @ self.prev_layer.nodes, self.biases)
        self.nodes = sig(self.zs)

class OutputLayer(HiddenLayer):
    def __init__(self, prev_layer, labels):
        super().__init__(len(labels), prev_layer)
        self.labels = labels
        self.result = None
        self.confidence = None

    def eval(self):
        super().eval()
        self.confidence = max(self.nodes)[0]
        self.result = self.labels[argmax(self.nodes)]
