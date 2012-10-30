import re
import numpy as np
import numpy.linalg as la
import features, filters

class Preprocessor(object):
    def __init__(self, pca = True):
        self.features = {}
        self.hiddenFeatures = {}
        self.filters = []
        self.pca = pca

    def _addHiddenFeature(self, f, key):
        if not f in self.hiddenFeatures:
            for requirement in f.requiredFeatures:
                self._addHiddenFeature(requirement)
            self.hiddenFeatures[f] = self.createFeature(f)

    def addFeature(self, f, key):
        for requirement in f.requiredFeatures:
            self._addHiddenFeature(requirement)
        self.features[key] = f
        
    def addFilter(self, f, key):
        for requirement in f.requiredFeatures:
            self._addHiddenFeature(requirement)
        self.filters[key] = f

    def createFeature(self, f):
        return self._createInstance(f, features)

    def createFilter(self, f):
        return self._createInstance(f, filters)

    def processData(self, data):
        #create features
        featureMap = {}
        for key in self.hiddenFeatures:
            featureMap[key] = self.hiddenFeatures[key].calc(data, featureMap).flatten()
        for key in self.features:
            featureMap[key] = self.features[key].calc(data, featureMap).flatten()
        #filter data
        featureMap['_filter_'] = np.array(np.zeros(len(featureMap[featureMap.keys()[0]])))
        for key in self.filters:
            self.filters[key].applyFilter(featureMap)
        outputLayers = [featureMap['_filter_']]
        for key in featureMap:
            if key != '_filter_' and key in self.features:
                outputLayers.append(featureMap[key])
        result = np.vstack(filter(lambda x : x[0] == 0, np.dstack(outputLayers)[0]))
        result = result[:, 1:]

        if self.pca:
            return self._calcPrincipleComponents(result)
        else:
            return result

    def _calcPrincipleComponents(self, data):

    def _createInstance(self, f, lib):
        m = re.match(r"(\w+)\((.*)\)", f)
        fargs = filter(None, m.group(2).split(','))
        fClass = getattr(lib, m.group(1))
        return fClass(*fargs)


