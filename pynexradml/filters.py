import numpy as np

BADVAL = 0x20000
RFVAL = BADVAL - 1

def minRange20km(dataLayers):
    data = dataLayers['range']
    for i in xrange(len(data)):
        if data[i] < 20000:
            dataLayers['_filter_'][i] = 1

def maxRange145km(dataLayers):
    data = dataLayers['range']
    for i in xrange(len(data)):
        if data[i] > 145000:
            dataLayers['_filter_'][i] = 1

def noBadRef(dataLayers):
    data = dataLayers['ref']
    for i in xrange(len(data)):
        if data[i] == BADVAL:
            dataLayers['_filter_'][i] = 1

def noBadVel(dataLayers):
    data = dataLayers['vel']
    for i in xrange(len(data)):
        if data[i] == BADVAL:
            dataLayers['_filter_'][i] = 1

def subSample(dataLayers):
    data = dataLayers['_filter_']
    for i in xrange(len(data)):
        if np.random.rand() > .1:
            data[i] = 1

class Filter(object):
    def __init__(self, name, requiredFeatures, func):
        self.name = name
        self.requiredFeatures = requiredFeatures
        self.applyFilter = func

class FilterLoader(object):
    def __init__(self):
        self.filterMap = {}
        self.filterMap['min_range_20km'] = Filter('min_range_20km', ['range'], minRange20km)
        self.filterMap['max_range_145km'] = Filter('max_range_145km', ['range'], maxRange145km)
        self.filterMap['no_bad_ref'] = Filter('no_bad_ref', ['ref'], noBadRef)
        self.filterMap['no_bad_vel'] = Filter('no_bad_vel', ['vel'], noBadVel)
        self.filterMap['sub'] = Filter('sub', [], subSample)

    def load(self, filterName):
        return self.filterMap[filterName]
