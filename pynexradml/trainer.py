from __future__ import division
import random
import numpy as np
import ffnn
import preprocessor
import datastore

def printStats(tp, tn, fp, fn, mse):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

    print "---------------------------------"
    print "Classifier Performance:"
    print ""
    print "  Confusion Matrix"
    print "       TP    |    FP     " 
    print "  +---------------------+"
    print "  |%10d|%10d|" % (tp, fp)
    print "  |%10d|%10d|" % (fn, tn)
    print "  +---------------------+"
    print "       FN    |    TN     " 
    print ""
    print "  - Accuracy = %.3f" % accuracy
    print "  - Precision = %.3f" % precision
    print "  - Recall = %.3f" % recall
    print "  - F1 = %.3f" % f1
    print "  - MSE = %.3f" % mse
    print "---------------------------------"

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description='Build and validate classifiers')
    parser.add_argument('-d', '--data_dir', help='Directory containing datastore')

    parser.add_argument('--epochs', type=int, help='Number of epochs for training a neural network')
    parser.add_argument('--features', nargs='*', help='Features to include (e.g. ref, vel, sw)')
    parser.add_argument('--filters', nargs='*', help='Filters to include (e.g. min_range_20km, no_bad_ref, su)')
    parser.add_argument('--norm', nargs='*', help='Normalizers to use (e.g. SymmetricNormalizer(0,1,2))')
    parser.add_argument('-t', '--training_data', nargs='*', help='Datasets to use for training data (e.g. rd1 rd2)')
    parser.add_argument('-o', '--output', help='Save classifier to an output file (e.g. nn.ml)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')

    args = parser.parse_args()

    ds = datastore.Datastore(args.data_dir)

    #Get Data
    sweeps = []
    for dataset in args.training_data:
        sweeps += [(dataset, x[0], x[1]) for x in ds.getManifest(dataset)]

    processor = preprocessor.Preprocessor()
    for f in args.features:
        print "Adding Feature %s to Preprocessor" % f
        processor.createAndAddFeature(f)
    for f in args.filters:
        print "Adding Filter %s to Preprocessor" % f
        processor.createAndAddFilter(f)
    for n in args.norm:
        print "Adding Normalizer %s to Preprocessor" % n
        processor.createAndAddNormalizer(n)

    instances = []
    print "Loading Data..."
    for sweep in sweeps:
        print "Constructing Data for %s, Class = %s" % (sweep[1], sweep[2])
        instance = processor.processData(ds.getData(sweep[0], sweep[1])) 
        instances.append(np.hstack([instance, np.ones((instance.shape[0], 1)) * float(sweep[2])]))

    composite = np.vstack(instances)
    print "Normalizing Data..."
    data = processor.normalizeData(composite)

    np.random.shuffle(data)

    validationIndex = int(len(data) * 0.9)

    nodes = len(data[0]) - 1
    network = ffnn.FeedForwardNeuralNet(layers=[nodes, nodes // 2, 1])
    network.learning_data = data[:validationIndex,:]
    network.validation_data = data[validationIndex:,:]
    network.shuffle = True
    network.momentum = 0.1
    network.verbose = args.verbose
    print "Learning Network..."
    network.learn(0.03, args.epochs)

    tp, tn, fp, fn, count = (0, 0, 0, 0, 0)
    def callback(inputs, target, output):
        global tp
        global tn
        global fp
        global fn
        global count

        if output > 0:
            output = 1
        elif output <= 0:
            output = -1

        target = int(target)

        if target == 1 and output == 1:
            tp += 1
        elif target == 1 and output != 1:
            fn += 1
        elif target == -1 and output == -1:
            tn += 1
        elif target == -1 and output != -1:
            fp += 1
        count += 1

    mse = network.validate(callback)
    printStats(tp, tn, fp, fn, mse)

    if args.output != None:
        print "Saving Network to %s" % (args.output + ".net")
        network.save(args.output + ".net")
        print "Saving Preprocessor to %s" % (args.output + ".proc")
        processor.save(args.output + ".proc")
