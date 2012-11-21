import os, tables

class Cache(object):
    def __init__(self, path="")
        self.path = path
        self.hfile = tables.openFile(os.path.join(self.path, "cache.h5"), mode="w")

    def createDiskArray(self, name, columns):
        ttype = tables.Float32Atom()
        return self.hfile.createEArray(hfile.root, name, ttype, (0, columns), name)

    def close(self):
        if self.hfile != None:
            self.hfile.close()
