import numpy as np

BADVAL = 0x20000
RFVAL = 0x1FFFF

class Filter(object):
    def __init__(self):
        self.requiredFeatures = []

class RangeConstraints(Filter):
    def __init__(self, minRange=20000, maxRange=145000):
        super(RangeConstraints, self).__init__()
        self.minRange = int(minRange)
        self.maxRange = int(maxRange)
        self.requiredFeatures.append('Range()')

    def apply(self, fmap):
        data = fmap['Range()']
        for i in xrange(len(data)):
            if (data[i] < self.minRange) or (data[i] > self.maxRange):
                fmap['_filter_'][i] = 1

class RemoveBadValues(Filter):
    def __init__(self, baseFeature, badValue=True, rangeFolded=True):
        super(RemoveBadValues, self).__init__()
        baseFeature += '()'
        self.bad = (badValue and (badValue != "False"))
        self.rf = (rangeFolded and (rangeFolded != "False"))
        self.baseFeature = baseFeature
        self.requiredFeatures.append(baseFeature)

    def apply(self, fmap):
        data = fmap[self.baseFeature]
        for i in xrange(len(data)):
            if (self.bad and data[i] == BADVAL) or (self.rf and data[i] == RFVAL):
                fmap['_filter_'][i] = 1

class SubSample(Filter):
    def __init__(self, samplePercent):
        super(SubSample, self).__init__()
        self.threshold = float(samplePercent)

    def apply(self, fmap):
        data = fmap['_filter_']
        for i in xrange(len(data)):
            if np.random.rand() > self.threshold:
                data[i] = 1

