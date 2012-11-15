import numpy as np

class Normalizer(object):
    def __init__(self, columns):
        self.columns = columns

class SymmetricNormalizer(Normalizer):
    def __init__(self, *columns):
        super(SymmetricNormalizer, self).__init__(columns)

    def apply(self, data):
        for index in [int(x) for x in self.columns]:
            data[:, index] -= np.mean(data[:, index])
            dev = np.std(data[:, index])
            data[:, index] = data[:, index] / dev if dev != 0 else data[:, index] 
