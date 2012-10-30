import argparse, os, re
from preprocessor import Preprocessor

parser = argparse.ArgumentParser(description='Build and validate classifiers')
parser.add_argument('--features', nargs='*', help='Features to include (e.g. ref, vel, sw)')
args = parser.parse_args()

processor = Preprocessor()
for f in args.features:
    feature = processor.createFeature(f)
    print feature.calc(None)


