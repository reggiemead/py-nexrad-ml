import os, sys
import numpy as np

sys.path.append("../")
sys.path.append("../../")

import unittest2
from pynexrad.classifier import L2NeuralNet
from pynexrad.classifier import NNClassifierBuilder

class TestClassifier(unittest2.TestCase):
    def setUp(self):
        self.net = L2NeuralNet()
        

    def test_load_features(self):
        self.net.loadFeatures(['ref', 'vel'])
        self.assertTrue('ref' in self.net.features)
        self.assertTrue('vel' in self.net.features)

    def test_build_features(self):
        self.net.loadFeatures(['ref', 'vel', 'sw'])
        layers = self.net.buildFeatures(np.array([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]], dtype='f'))
        self.assertTrue('ref' in layers)
        self.assertTrue('vel' in layers)
        self.assertTrue('sw' in layers)

        self.assertTrue(layers['ref'].shape == (4,))
        self.assertTrue(layers['vel'].shape == (4,))
        self.assertTrue(layers['sw'].shape == (4,))

        self.assertTrue(sum(layers['ref']) == 0)
        self.assertTrue(sum(layers['vel']) == 4)
        self.assertTrue(sum(layers['sw']) == 8)

    def test_load_filters(self):
        self.net.loadFilters(['no_bad_ref', 'no_bad_vel'])

    def test_kFoldCrossValidation(self):
        builder = NNClassifierBuilder(None, None)
        for folds in [3, 5, 10]:
            for num in [10, 20, 44, 83, 100]:
                inputs = range(num)
                fold_size = len(inputs) // folds
                inputs = inputs[0:fold_size*folds]
                for train, test in builder.kFoldCrossValidation(folds, inputs):
                    self.assertEqual(len(train), fold_size * (folds - 1))
                    self.assertEqual(len(test), fold_size)
"""
    def test_example(self):
        self.assertEqual(len([]), 0)
        self.assertRaises(TypeError, random.shuffle, (1,2,3))
        self.assertTrue(element in self.seq)
"""

if __name__ == '__main__':
    suite = unittest2.TestLoader().loadTestsFromTestCase(TestClassifier)
    unittest2.TextTestRunner(verbosity=2).run(suite)
