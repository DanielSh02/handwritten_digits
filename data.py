import scipy.io
import scipy.stats
from scipy.special import softmax

class DataSet(list):
    def __init__(self, raw):
        self.data = raw[0, 0]
        self.idx = 0
        self.input_len = len(self.data['images'][0, 0][0])

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data['images'][0, 0])

    def __next__(self):
        self.idx += 1
        try:
            return self.data['images'][0, 0][self.idx - 1] / 255, self.data['labels'][0, 0][self.idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration

    def shuffle(self):
        shuffle(self.data)
