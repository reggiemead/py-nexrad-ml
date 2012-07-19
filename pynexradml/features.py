import numpy as np
import scipy.stats as stats

BADVAL = 0x20000
RFVAL = BADVAL - 1

def normalize(data):
    dmin = np.min(data)
    dmax = np.max(data)
    result = 2 * ((data - dmin) / (dmax - dmin)) - 1
    return result

#features 
def calcRef(data):
    return data[:,:,0]

def calcNRef(data):
    return normalize(calcRef(data))

def calcVel(data):
    return data[:,:,1]

def calcNVel(data):
    return normalize(calcVel(data))

def calcSW(data):
    return data[:,:,2]

def calcNSW(data):
    return normalize(calcSW(data))

def calcRange(data):
    result = np.array(np.zeros((360, 1840)))
    for az in xrange(360):
        for r in xrange(1840):
            result[az, r] = r * 250
    return result

def calcSkew(data, window = 11):
    border = window // 2
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

def calcKurtosis(data, window = 3):
    border = window // 2
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

def calcVariance(data, window = 3):
    border = window // 2
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

def calcRefVariance(data):
    return calcVariance(data[:,:,0])

def calcVelVariance(data):
    return calcVariance(data[:,:,0])

def calcSwVariance(data):
    return calcVariance(data[:,:,0])

def calcRefSkew(data):
    return calcSkew(data[:,:,0])

def calcVelSkew(data):
    return calcSkew(data[:,:,0])

def calcSwSkew(data):
    return calcSkew(data[:,:,0])

def calcRefKurtosis(data):
    return calcKurtosis(data[:,:,0])

def calcVelKurtosis(data):
    return calcKurtosis(data[:,:,0])

def calcSwKurtosis(data):
    return calcKurtosis(data[:,:,0])

class Feature(object):
    def __init__(self, name, function):
        self.name = name
        self.build = function

class FeatureLoader(object):
    def __init__(self):
        self.featureMap = {
            "ref" : calcRef,
            "vel" : calcVel,
            "sw" : calcSW,
            "nref" : calcNRef,
            "nvel" : calcNVel,
            "nsw" : calcNSW,
            "range" : calcRange,
            "ref_var" : calcRefVariance,
            "vel_var" : calcVelVariance,
            "sw_var" : calcSwVariance,
            "ref_kurt" : calcRefKurtosis,
            "vel_kurt" : calcVelKurtosis,
            "sw_kurt" : calcSwKurtosis,
            "ref_skew" : calcRefSkew,
            "vel_skew" : calcVelSkew,
            "sw_skew" : calcSwSkew,
        }

    def load(self, featureName):
        return Feature(featureName, self.featureMap[featureName])
