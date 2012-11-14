import random
import numpy as np
import ffnn
import preprocessor
import datatore

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(description='Build and validate classifiers')
    parser.add_argument('-d', '--data_dir', help='Directory containing datastore')

    parser.add_argument('--epochs', type=int, help='Number of epochs for training a neural network')
    parser.add_argument('--features', nargs='*', help='Features to include (e.g. ref, vel, sw)')
    parser.add_argument('--filters', nargs='*', help='Filters to include (e.g. min_range_20km, no_bad_ref, su)')
    parser.add_argument('-t', '--training_data', nargs='*', help='Datasets to use for training data (e.g. rd1 rd2)')

    args = parser.parse_args()

    ds = datastore.Datastore(args.data_dir)

    #Get Data
    data = []

    processor = preprocessor.Preprocessor()
    for f in args.features:
        processor.createAndAddFeature(f)
    for f in args.filters:
        processor.createAndAddFilter(f)

    #data = processor.processData(data)
    data = processor.calcPCA(data)
    np.random.shuffle(data)

    validationIndex = int(len(data) * 0.9)

    nodes = len(data[0]) - 1
    network = ffnn.FeedForwardNeuralNet(layers=[nodes, nodes // 2, 1])
    network.learning_data = data[:validationIndex,:]
    network.validation_data = data[validationIndex:,:]
    network.shuffle = True
    network.momentum = 0.1
    network.learn(0.03, args.epochs)
    network.validate()
