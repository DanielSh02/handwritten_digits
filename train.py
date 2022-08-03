import sys
from network import *
import scipy.io
from os.path import exists
from pickle import load


train_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['train']
test_set = scipy.io.loadmat('matlab/emnist-mnist.mat')['dataset']['test']

def train(name, n):
    file = f"pickles/{name}.pickle"
    if exists(file):
        with open(file, "rb") as f:
            net = load(f)
    else:
        layers = [100, 100]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        net = Network(layers, labels, train_set)
    net.train(n)
    print(f"Testing trained network")
    net.test(test_set)
    net.dump(name)

if __name__ == '__main__':
    train("100-20", 10)