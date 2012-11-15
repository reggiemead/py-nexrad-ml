import numpy as np

class Normalizer(object):
    def __init__(self, columns):
        self.columns = columns

class SymmetricNormalizer(Normalizer):
    def __init__(self, *columns):
        super(SymmetricNormalizer, self).__init__(columns)
        self.means = {}
        self.devs = {}

    def serialize(self):
        properties = {
            'means' : self.means,
            'devs' : self.devs
        }
        return properties

    def deserialize(self, properties):
        self.means = properties['means']
        self.devs = properties['devs']

    def applyAgain(self, data):
        for index in [int(x) for x in self.columns]:
            data[:, index] -= self.means[str(index)]
            dev = self.devs[str(index)]
            data[:, index] = data[:, index] / dev if dev != 0 else data[:, index] 

    def apply(self, data):
        for index in [int(x) for x in self.columns]:
            self.means[index] = np.mean(data[:, index]) 
            data[:, index] -= self.means[index]
            dev = np.std(data[:, index])
            self.devs[index] = dev
            data[:, index] = data[:, index] / dev if dev != 0 else data[:, index] 
