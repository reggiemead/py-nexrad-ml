from __future__ import division

import os,sys,level2,glob
import pyximport
pyximport.install()

from util import LOG
from fast_util import resample_sweep_polar
import ffnn, preprocessor
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load Classifier and Screen new Data')
    parser.add_argument('-i', '--input', help='Name of classifer to load e.g. myclassifier')
    parser.add_argument('-d', '--data', help='Location of data to screen')
    parser.add_argument('-t', '--threshold', type=float, help='Pulse volume threshold ratio e.g. 0.7')
    args = parser.parse_args()

    sweeps = glob.glob(os.path.join(args.data, "*.Z")) + glob.glob(os.path.join(args.data, "*.gz"))

    net = ffnn.FeedForwardNeuralNet.load(args.input + ".net")
    processor = preprocessor.Preprocessor.load(args.input + ".proc")

    for sweep in sweeps:
        data = processor.normalizeAdditionalData(processor.processData(resample_sweep_polar(level2.Sweep(sweep))))
        hits = 0
        for datum in data:
            output = net.activate(datum)
            LOG("Activation = %f" % output)
            if output > 0:
                hits += 1
        ratio = hits / len(data);
        if (ratio > args.threshold):
            print "%s MATCHES screen criteria (ratio = %f, threshold = %f)" % (sweep, ratio, args.threshold)
        else:
            print "%s - not a match (ratio = %f, threshold = %f)" % (sweep, ratio, args.threshold)


