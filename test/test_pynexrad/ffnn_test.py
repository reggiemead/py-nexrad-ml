import os, sys

sys.path.append("../")
sys.path.append("../../")

import numpy as np
import unittest2 as unittest
from pynexradml.ffnn import FeedForwardNeuralNet
from pynexradml.ffnn import LogisticActivationFunction
from pynexradml.ffnn import HyperbolicTangentActivationFunction

class TestFFNN(unittest.TestCase):
    def setUp(self):
        pass

    def testTooFewLayers(self):
        self.assertRaises(ValueError, FeedForwardNeuralNet, ([3]))

    def testWeightMatrixShape(self):
        layers = [3, 2, 1]
        net = FeedForwardNeuralNet(layers)
        self.assertEqual(len(net.weights), 2)
        self.assertEqual(net.weights[0].shape, (4,2))
        self.assertEqual(net.weights[1].shape, (3,1))

    def testHTanSigmoid(self):
        sigmoid = HyperbolicTangentActivationFunction(1, 1)
        output = sigmoid.activate(np.matrix([-1, 0, 1]))
        self.assertAlmostEqual(output[0,0], -.761594, places=5)
        self.assertAlmostEqual(output[0,1], .000000, places=5)
        self.assertAlmostEqual(output[0,2], .761594, places=5)

    def testScaledHTanSigmoid(self):
        sigmoid = HyperbolicTangentActivationFunction(1.7159, .666666)
        output = sigmoid.activate(np.matrix([-1, 0, 1]))
        self.assertAlmostEqual(output[0,0], -1.0, places=5)
        self.assertAlmostEqual(output[0,1], 0.0, places=5)
        self.assertAlmostEqual(output[0,2], 1.0, places=5)

    def testScaledHTanSigmoidDerivative(self):
        sigmoid = HyperbolicTangentActivationFunction(1.7159, .666666)
        output = sigmoid.derivative(np.matrix([-1.715899, 0, 1.715899]))
        self.assertAlmostEqual(output[0,0], 0, places=4)
        self.assertAlmostEqual(output[0,1], 1.1439, places=4)
        self.assertAlmostEqual(output[0,2], 0, places=4)

    def testLogisticSigmoid(self):
        sigmoid = LogisticActivationFunction()
        output = sigmoid.activate(np.matrix([-1, -.5, 0, .5, 1]))
        self.assertAlmostEqual(output[0,0], .268941, places=5)
        self.assertAlmostEqual(output[0,1], .377540, places=5)
        self.assertAlmostEqual(output[0,2], .500000, places=5)
        self.assertAlmostEqual(output[0,3], .622459, places=5)
        self.assertAlmostEqual(output[0,4], .731058, places=5)

    def testLogisticSigmoidDerivative(self):
        sigmoid = LogisticActivationFunction()
        output = sigmoid.derivative(np.matrix([.25, .5, .75]))
        self.assertAlmostEqual(output[0,0], .1875, places=4)
        self.assertAlmostEqual(output[0,1], .2500, places=4)
        self.assertAlmostEqual(output[0,2], .1875, places=4)

    def testActivation(self):
        net = FeedForwardNeuralNet([3, 2, 1], sigmoids=[LogisticActivationFunction(), LogisticActivationFunction()])
        net.weights[0] = np.matrix([[1 , 2], [1, 2], [-1, -2], [-1, -2]])
        net.weights[1] = np.matrix([[-1],[-1],[1]])
        output = net.activate([1, 1, 1])[0,0]
        self.assertAlmostEqual(output, .5, places=5)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFFNN)
    unittest.TextTestRunner(verbosity=2).run(suite)
