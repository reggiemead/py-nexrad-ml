from __future__ import division

import os,sys,level2,glob
import pyximport
pyximport.install()

from util import LOG
from fast_util import resample_sweep_polar
from classifier import L2NeuralNet
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Load Classifier and Screen new Data')
    parser.add_argument('-c', '--classifier')
    parser.add_argument('-d', '--data')
    parser.add_argument('-t', '--threshold', type=float)
    args = parser.parse_args()

    sweeps = glob.glob(os.path.join(args.data, "*.Z")) + glob.glob(os.path.join(args.data, "*.gz"))

    net = L2NeuralNet()
    net.load(args.classifier)

    for sweep in sweeps:
        data = net.transformData(resample_sweep_polar(level2.Sweep(sweep)))
        hits = 0
        for datum in data:
            output = net.activate(datum)
            LOG("Activation = %f" % output)
            if output > 0.5:
                hits += 1
        ratio = hits / len(data);
        if (ratio > args.threshold):
            print "%s matches screen criteria (ratio = %f, threshold = %f)" % (sweep, ratio, args.threshold)
        else:
            LOG("%s - NOT A MATCH (ratio = %f, threshold = %f)" % (sweep, ratio, args.threshold))


