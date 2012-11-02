import re
import numpy as np
import numpy.linalg as la
import features, filters

class Preprocessor(object):
    def __init__(self, pca = False):
        self.features = {}
        self.featureKeys = []
        self.hiddenFeatures = {}
        self.hiddenFeatureKeys = []
        self.filters = {}
        self.filterKeys = []
        self.pca = pca

    def _addHiddenFeature(self, f):
        if not f in self.hiddenFeatures:
            feature = self.createFeature(f)
            for requirement in feature.requiredFeatures:
                self._addHiddenFeature(requirement)
            self.hiddenFeatureKeys.append(f)
            self.hiddenFeatures[f] = feature

    def addFeature(self, f, key):
        for requirement in f.requiredFeatures:
            self._addHiddenFeature(requirement)
        self.featureKeys.append(key)
        self.features[key] = f
        
    def addFilter(self, f, key):
        for requirement in f.requiredFeatures:
            self._addHiddenFeature(requirement)
        self.filterKeys.append(key)
        self.filters[key] = f

    def createFeature(self, f):
        return self._createInstance(f, features)

    def createFilter(self, f):
        return self._createInstance(f, filters)

    def createAndAddFeature(self, f):
        self.addFeature(self.createFeature(f), f)

    def createAndAddFilter(self, f):
        self.addFilter(self.createFilter(f), f)

    def processData(self, data):
        #create features
        featureMap = {}
        for key in self.hiddenFeatureKeys:
            featureMap[key] = self.hiddenFeatures[key].calc(data, featureMap)
        for key in self.featureKeys:
            if not key in featureMap:
                featureMap[key] = self.features[key].calc(data, featureMap)

        for key in featureMap:
            featureMap[key] = featureMap[key].flatten()

        #filter data
        featureMap['_filter_'] = np.array(np.zeros(len(featureMap[featureMap.keys()[0]])))
        for key in self.filterKeys:
            self.filters[key].apply(featureMap)
        outputLayers = [featureMap['_filter_']]
        for key in self.featureKeys:
            if key in self.features:
                outputLayers.append(featureMap[key])
        result = np.vstack(filter(lambda x : x[0] == 0, np.dstack(outputLayers)[0]))
        result = result[:, 1:]

        if self.pca:
            return self.calcPCA(result)
        else:
            return result

    def calcPCA(self, data):
        data -= np.mean(data, axis=0)
        #data = data / np.std(data, axis=0)
        c = np.cov(data, rowvar=0)
        values, vectors = la.eig(c)
        featureVector = vectors[:, [values.tolist().index(x) for x in np.sort(values)[::-1]]]
        return (np.matrix(featureVector) * np.matrix(data.T)).T

    def _createInstance(self, f, lib):
        m = re.match(r"(\w+)\((.*)\)", f)
        fargs = [x.strip() for x in filter(None, m.group(2).split(','))]
        fClass = getattr(lib, m.group(1))
        return fClass(*fargs)


