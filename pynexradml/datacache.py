import tables, hashlib
import numpy as np
import util

class Datacache(object):
    def __init__(self, cacheName):
        self.cacheName = cacheName

    def computeHash(self, features, filters):
        hashCode = hashlib.md5()
        hashCode.update(':'.join(features) + '&' + ':'.join(filters))
        return hashCode.hexdigest()

    def hasData(self, name, features, filters):
        with tables.openFile(self.cacheName, 'a') as cache:
            hashCode = self.computeHash(features, filters)
            hashCode = "HASH%s" % hashCode
            util.LOG("Hashcode = %s" % hashCode)
            return (hashCode + '/' + name) in cache.root

    def getData(self, name, features, filters):
        with tables.openFile(self.cacheName, 'a') as cache:
            hashCode = self.computeHash(features, filters)
            hashCode = "HASH%s" % hashCode
            return np.array(cache.getNode('/' + hashCode, name))

    def setData(self, data, name, features, filters):
        with tables.openFile(self.cacheName, 'a') as cache:
            hashCode = self.computeHash(features, filters)
            hashCode = "HASH%s" % hashCode
            if not hashCode in cache.root:
                cache.createGroup("/", hashCode)
            atom = tables.Float32Atom()
            placeholder = cache.createCArray('/' + hashCode, name, atom, data.shape)
            placeholder[:] = data
