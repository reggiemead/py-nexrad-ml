import datastore, datacache, util
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from ffnn import FeedForwardNeuralNet
from features import FeatureLoader
from filters import FilterLoader

class L2NeuralNet(FeedForwardNeuralNet):
    def __init__(self, i_count=3, h_count=3):
        super(L2NeuralNet, self).__init__(i_count, h_count)
        self.i_count = i_count
        self.datastore = None
        self.datacache = None
        self.training_sweeps = []
        self.validation_sweeps = []
        self.features = {}
        self.filters = {}
        self.filterFeatures = {}
        self.memcache = {}
        self.sweepThreshold = None

    def serialize(self):
        properties = super(L2NeuralNet, self).serialize()
        properties['features'] = ','.join(self.features.keys())
        properties['filters'] = ','.join(self.filters.keys())
        return properties

    def deserialize(self, properties):
        super(L2NeuralNet, self).deserialize(properties)
        self.loadFeatures(properties['features'].split(','))
        self.loadFilters(properties['filters'].split(','))

    def loadFeatures(self, features):
        fl = FeatureLoader()
        for fname in features:
            self.features[fname] = fl.load(fname)

    def loadFilters(self, filters):
        loader1 = FilterLoader()
        loader2 = FeatureLoader()
        for fname in filters:
            flt = loader1.load(fname)
            for feature in flt.requiredFeatures:
                if not (feature in self.features or feature in self.filterFeatures):
                    self.filterFeatures[feature] = loader2.load(feature)
            self.filters[fname] = flt 

    def buildFeatures(self, data):
        layers = {}
        for fname in self.features:
            layers[fname] = self.features[fname].build(data).flatten()
        for fname in self.filterFeatures:
            layers[fname] = self.filterFeatures[fname].build(data).flatten()
        return layers 

    def filterData(self, layers):
        layers['_filter_'] = np.array(np.zeros(len(layers[layers.keys()[0]])))
        for key in self.filters:
            self.filters[key].applyFilter(layers)
        outputLayers = [layers['_filter_']]
        for key in layers:
            if key != '_filter_' and key in self.features:
                outputLayers.append(layers[key])
        result = np.vstack(filter(lambda x : x[0] == 0, np.dstack(outputLayers)[0]))
        return result[:, 1:]

    def transformData(self, data):
        layers = self.buildFeatures(data)
        return self.filterData(layers)

    def getCachedData(self, name, dataset, target):
        features = self.features.keys()
        filters = self.filters.keys()
        if name in self.memcache:
            util.LOG("Retrieving %s from memcache, target = %.1f" % (name, target))
            data = self.memcache[name]
        elif self.datacache.hasData(name, features, filters):
            util.LOG("Retrieving %s from diskcache, target = %.1f" % (name, target))
            data = self.datacache.getData(name, features, filters)
            self.memcache[name] = data
        else:
            util.LOG("Caching %s, target = %.1f" % (name, target))
            data = self.transformData(self.datastore.getData(dataset, name))
            self.datacache.setData(data, name, features, filters)
            self.memcache[name] = data
        return data

    def get_data(self, sweeps):
        for sweep in sweeps:
            name = sweep[1]
            print "%s" % name
            target = float(sweep[2])
            for x in self.getCachedData(name, sweep[0], target):
                if x[0] != 0.0:
                    yield (x, target)

    def get_learning_data(self):
        print "Begin Learning"
        for inputs, target in self.get_data(self.training_sweeps):
            yield (inputs, target)

    def get_validation_data(self):
        print "Begin Validation"
        for inputs, target in self.get_data(self.validation_sweeps):
            yield (inputs, target)

    def validate(self):
        if self.sweepThreshold == None:
            super(L2NeuralNet, self).validate()
        else:
            print "Begin Validation"
            (mse, count, tp, tn, fp, fn) = 0, 0, 0, 0, 0, 0 
            self.activations = []
            for sweep in self.validation_sweeps:
                name = sweep[1]
                print "%s" % name
                target = float(sweep[2])
                hits = 0
                scount = 0
                for x in self.getCachedData(name, sweep[0], target):
                    output = self.activate(x)
                    self.activations.append(output)
                    mse += ((target - output)**2)
                    output = 1.0 if output > 0.5 else 0.0
                    if output == 1.0:
                        hits += 1
                    scount += 1

                count += scount
                exceedsThreshold = (hits / scount) > self.sweepThreshold
                if exceedsThreshold and target == 1.0:
                    tp += 1
                elif exceedsThreshold and target == 0.0:
                    fp += 1
                elif not exceedsThreshold and target == 0.0:
                    tn += 1
                elif not exceedsThreshold and target == 1.0:
                    fn += 1
            self.mse = mse / count
            self.printStats(tp, tp, fp, fn)
        return self.mse

class NNClassifierBuilder(object):
    def __init__(self, datastore, datacache):
        self.ds = datastore
        self.dc = datacache
        self.net = None
        self.bestmse = 0.0
        self.features = []
        self.filters = []
        self.epochs = 10
        self.learning = 0.1
        self.sweepThreshold = None

    def kFoldCrossValidation(self, k_folds, inputs):
        k_length = len(inputs) // k_folds
        for k in xrange(k_folds):
            train = [x for i, x in enumerate(inputs) if (i // k_length) != k]
            test = [x for i, x in enumerate(inputs) if (i // k_length) == k]
            yield (train, test)

    def build(self, training, folds):
        sweeps = []
        for dataset in training:
            sweeps += [(dataset, x[0], x[1]) for x in self.ds.getManifest(dataset)]
        random.shuffle(sweeps) 
        fold_size = len(sweeps) // folds
        sweeps = sweeps[0:fold_size*folds]
        fold = 0
        for training, validation in self.kFoldCrossValidation(folds, sweeps):
            fold += 1
            print "Fold %d" % fold
            network = L2NeuralNet(len(self.features), len(self.features)*2)
            network.datastore = self.ds
            network.datacache = self.dc
            network.loadFeatures(self.features)
            network.loadFilters(self.filters)
            network.training_sweeps = training
            network.validation_sweeps = validation
            network.sweepThreshold = self.sweepThreshold

            print "Learning Network Parameters"
            network.learn(learning=self.learning, epochs=self.epochs)
            mse = network.validate()
            if self.net == None or mse < self.bestmse:
                self.net = network
                self.bestmse = mse

            print "MSE = %s" % mse

            """
            plt.plot(range(1, len(network.accum_mse) + 1, 1), network.accum_mse)
            plt.xlabel('epochs')
            plt.ylabel('mean squared error')
            plt.grid(True)
            plt.title("Mean Squared Error by Epoch")
            plt.show()
            """

    def save(self, output_file):
        self.net.save(output_file)

    def load(self, input_file):
        self.net.load(input_file)

if __name__ == "__main__":
    import argparse, os


    parser = argparse.ArgumentParser(description='Build and validate classifiers')
    parser.add_argument('-b', '--build')
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('--features', nargs='*')
    parser.add_argument('--filters', nargs='*')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning', type=float, default=0.1)
    parser.add_argument('--sweep_threshold', type=float)
    parser.add_argument('-t', '--training_data', nargs='*')
    parser.add_argument('-o', '--output')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    util.debug = args.debug

    if args.build == 'neural_net':
        ds = datastore.Datastore(args.data_dir)
        dc = datacache.Datacache(os.path.join(args.data_dir, 'cache.h5'))
        builder = NNClassifierBuilder(ds, dc)
        builder.features = args.features
        builder.filters = args.filters
        builder.learning = args.learning
        builder.epochs = args.epochs
        if args.sweep_threshold != None:
            builder.sweepThreshold = args.sweep_threshold
        builder.build(args.training_data, args.folds)
        builder.save(args.output)
     else:
         parser.print_help()
