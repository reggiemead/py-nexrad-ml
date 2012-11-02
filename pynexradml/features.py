import numpy as np
import scipy.stats as stats

BADVAL = 0x20000
RFVAL = 0x1FFFF

class Feature(object):
    def __init__(self):
        self.requiredFeatures = []

class Reflectivity(Feature):
    def calc(self, data, fmap):
        return data[:,:,0]

class Velocity(Feature):
    def calc(self, data, fmap):
        return data[:,:,1]

class SpectrumWidth(Feature):
    def calc(self, data, fmap):
        return data[:,:,2]

class Range(Feature):
    def calc(self, data, fmap):
        result = np.array(np.zeros((360, 1840)))
        for az in xrange(360):
            for r in xrange(1840):
                result[az, r] = r * 250
        return result

class Variance(Feature):
    def __init__(self, baseFeature, window=3):
        super(Variance, self).__init__()
        baseFeature += '()'
        self.baseFeature = baseFeature
        self.window = window
        self.requiredFeatures.append(baseFeature)

    def calc(self, baseData, fmap):
        data = fmap[self.baseFeature]
        border = self.window // 2
        dim = data.shape

        ref = np.hstack([np.zeros((border*2 + dim[0], border)), np.vstack([np.zeros((border, dim[1])), data, np.zeros((border, dim[1]))]), np.zeros((border*2 + dim[0], border))])
        
        result = np.array(np.zeros(dim))
        for row in xrange(border, border + dim[0]):
            for col in xrange(border, border + dim[1]):
                tmp = ref[row - border : row + border + 1, col - border : col + border + 1]
                tmp = tmp - ((tmp == BADVAL) * BADVAL) #Set any badval in temporary matrix to zero
                tmp = tmp - ((tmp == RFVAL) * RFVAL) #Set any rfval in temporary matrix to zero
                result[row - border, col - border] = np.var(tmp)
        return result

class Kurtosis(Feature):
    def __init__(self, baseFeature, window=3):
        super(Kurtosis, self).__init__()
        baseFeature += '()'
        self.baseFeature = baseFeature
        self.window = window
        self.requiredFeatures.append(baseFeature)

    def calc(self, baseData, fmap):
        data = fmap[self.baseFeature]
        border = self.window // 2
        dim = data.shape
        ref = np.hstack([np.zeros((border*2 + dim[0], border)), np.vstack([np.zeros((border, dim[1])), data, np.zeros((border, dim[1]))]), np.zeros((border*2 + dim[0], border))])
        
        result = np.array(np.zeros(dim))
        for row in xrange(border, border + dim[0]):
            for col in xrange(border, border + dim[1]):
                tmp = ref[row - border : row + border + 1, col - border : col + border + 1]
                tmp = tmp - ((tmp == BADVAL) * BADVAL) #Set any badval in temporary matrix to zero
                tmp = tmp - ((tmp == RFVAL) * RFVAL) #Set any rfval in temporary matrix to zero
                result[row - border, col - border] = stats.kurtosis(tmp, axis=None)
        return result

class Skew(Feature):
    def __init__(self, baseFeature, window=3):
        super(Skew, self).__init__()
        baseFeature += '()'
        self.baseFeature = baseFeature
        self.window = window
        self.requiredFeatures.append(baseFeature)

    def calc(self, baseData, fmap):
        data = fmap[self.baseFeature]
        border = self.window // 2
        dim = data.shape

        ref = np.hstack([np.zeros((border*2 + dim[0], border)), np.vstack([np.zeros((border, dim[1])), data, np.zeros((border, dim[1]))]), np.zeros((border*2 + dim[0], border))])
        
        result = np.array(np.zeros(dim))
        for row in xrange(border, border + dim[0]):
            for col in xrange(border, border + dim[1]):
                tmp = ref[row - border : row + border + 1, col - border : col + border + 1]
                tmp = tmp - ((tmp == BADVAL) * BADVAL) #Set any badval in temporary matrix to zero
                tmp = tmp - ((tmp == RFVAL) * RFVAL) #Set any rfval in temporary matrix to zero
                result[row - border, col - border] = stats.skew(tmp, axis=None)
        return result
    

