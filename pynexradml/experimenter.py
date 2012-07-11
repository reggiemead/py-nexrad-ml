from __future__ import division

import numpy as np
import libxml2, os, tables, random
import level2
import matplotlib
import matplotlib.pyplot as plt

import pyximport
pyximport.install()

from fast_util import resample_sweep
from pyneurgen.neuralnet import NeuralNet


class L2NeuralNet(NeuralNet):
    def __init__(self):
        super(L2NeuralNet, self).__init__()
        self.datastore = None
        self.training_sweeps = []
        self.validation_sweeps = []
        self.test_sweeps = []

    def get_data(self, sweeps):
        for sweep in sweeps:
            target = [1.0] if sweep.startswith("/bio") else [0.0]
            data = self.datastore.getNode(sweep)
            for x in xrange(data.shape[0]):
                for y in xrange(data.shape[1]):
                    if data[x,y,0] != 0.0:
                        yield ([float(x) for x in data[x,y,:]], target)

    def get_learn_data(self, random_testing=None):
        for inputs, targets in self.get_data(self.training_sweeps):
            yield (inputs, targets)

    def get_validation_data(self):
        for inputs, targets in self.get_data(self.validation_sweeps):
            yield (inputs, targets)

    def get_test_data(self):
        for inputs, targets in self.get_data(self.test_sweeps):
            yield (inputs, targets)

def kFoldCrossValidation(k_folds, inputs):
    k_length = len(inputs) // k_folds
    for k in xrange(k_folds):
        train = [x for i, x in enumerate(inputs) if (i // k_length) != k]
        test = [x for i, x in enumerate(inputs) if (i // k_length) == k]
        yield (train, test)

def runNNExperiment(datastore, sweep_level=True, threshold=.7):
    ds = tables.openFile(datastore, "a")
    sweeps = ['/bio/' + x.name for x in ds.root.bio] + ['/nonbio/' + x.name for x in ds.root.nonbio]
    random.shuffle(sweeps)
    if sweep_level:
        fold_size = len(sweeps) // 10
        sweeps = sweeps[0:fold_size*10]

    #generate data
    for training, validation in kFoldCrossValidation(10, sweeps):
        print "Testing Next Fold"
        #construct network
        net = L2NeuralNet()
        net.init_layers(3, [9], 1)
        net.randomize_network()
        net.set_halt_on_extremes(True)
        net.set_random_constraint(.5)
        net.set_learnrate(.1)

        net.datastore = ds
        net.training_sweeps = training
        net.validation_sweeps = validation
        net.test_sweeps = validation

        print "Learning Network Parameters"
        net.learn(epochs=100, show_epoch_results=True, random_testing=False)
        mse = net.test()

        plt.plot(range(1, len(net.accum_mse) + 1, 1), net.accum_mse)
        plt.xlabel('epochs')
        plt.ylabel('mean squared error')
        plt.grid(True)
        plt.title("Mean Squared Error by Epoch")

        plt.show()

def rebuildPreprocessedDataCache(datastore):
    ds = tables.openFile(datastore, "a")
    if 'cache' in ds.root:
        ds.removeNode("/", "cache", True)
    ds.createGroup("/", "cache", "PreProcessed Data Cache")
    """
    TODO
    """

def importData(datastore, path):
    ds = tables.openFile(datastore, "a")

    if not 'bio' in ds.root:
        ds.createGroup("/", "bio", "Biological Data")
    if not 'nonbio' in ds.root:
        ds.createGroup("/", "nonbio", "Nonbiological Data")

    doc = libxml2.parseFile(path + "/classes.xml")
    ctxt = doc.xpathNewContext()
    biofiles = [(x.getContent()) for x in ctxt.xpathEval("/training_sweeps/sweep[@class=1]/@id")]
    nonbiofiles = [(x.getContent()) for x in ctxt.xpathEval("/training_sweeps/sweep[@class=0]/@id")]

    for bfile in biofiles:
        print("Checking %s" % (bfile))
        (fname, fext) = os.path.splitext(bfile)
        if fname.startswith("6500"):
            fname = fname[4:]
        if not "/bio/" + fname in ds:
            loadScanIntoDataset(ds, "/bio", path + "//" + bfile, fname)
    for nfile in nonbiofiles:
        print("Checking %s" % (nfile))
        (fname, fext) = os.path.splitext(nfile)
        if fname.startswith("6500"):
            fname = fname[4:]
        if not "/nonbio/" + fname in ds:
            loadScanIntoDataset(ds, "/nonbio", path + "//" + nfile, fname)
    ds.close()

def loadScanIntoDataset(ds, where, sname, nname):
    print("Loading %s into DataStore [class = %s]" % (nname, where))
    data = resample_sweep(level2.Sweep(sname))

    atom = tables.Float32Atom()
    filters = tables.Filters(complib='lzo', complevel=5)
    vscan = ds.createCArray(where, nname, atom, data.shape, filters=filters)
    vscan[:,:,:] = data
