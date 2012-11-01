from __future__ import division
import os, sys
import numpy as np

sys.path.append("../")
sys.path.append("../../")
import unittest2 as unittest
import pynexradml.preprocessor as preprocessor

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.processor = preprocessor.Preprocessor()
        pass
        
    def testReflectivity(self):
        feature = self.processor.createFeature("Reflectivity()")
        data = np.array([[[1, 0, 0], [2, 0, 0]],
                         [[3, 0, 0], [4, 0, 0]]])
        output = feature.calc(data, {}) 
        self.assertTrue(np.allclose(output.flatten(), np.array([1, 2, 3, 4])))

    def testVelocity(self):
        feature = self.processor.createFeature("Velocity()")
        data = np.array([[[1, 5, 0], [2, 6, 0]],
                         [[3, 7, 0], [4, 8, 0]]])
        output = feature.calc(data, {}) 
        self.assertTrue(np.allclose(output.flatten(), np.array([5, 6, 7, 8])))

    def testSpectrumWidth(self):
        feature = self.processor.createFeature("SpectrumWidth()")
        data = np.array([[[1, 5, 9], [2, 6, 10]],
                         [[3, 7, 11], [4, 8, 12]]])
        output = feature.calc(data, {}) 
        self.assertTrue(np.allclose(output.flatten(), np.array([9,10,11,12])))

    def testVariance(self):
        feature = self.processor.createFeature("Variance(Reflectivity)")
        self.assertEqual(feature.requiredFeatures[0], "Reflectivity()")

        data = np.array([[1, 2, 3],
                         [4, 5, 6], 
                         [7, 8, 9]])

        fmap = {"Reflectivity()" : data}
        output = feature.calc(None, fmap) 
        self.assertEqual(output.shape, (3,3))
        self.assertAlmostEqual(output[0,0], 3.333333, places=5)
        self.assertAlmostEqual(output[1,1], 6.666666, places=5)

    def testSkew(self):
        feature = self.processor.createFeature("Skew(Reflectivity)")
        self.assertEqual(feature.requiredFeatures[0], "Reflectivity()")

        data = np.array([[1, 2, 3],
                         [1, 5, 6], 
                         [1, 8, 9]])

        fmap = {"Reflectivity()" : data}
        output = feature.calc(None, fmap) 
        self.assertEqual(output.shape, (3,3))
        self.assertAlmostEqual(output[1,1], 0.470330, places=5)

    def testKurtosis(self):
        feature = self.processor.createFeature("Kurtosis(Reflectivity)")
        self.assertEqual(feature.requiredFeatures[0], "Reflectivity()")

        data = np.array([[1, 2, 3],
                         [4, 5, 6], 
                         [7, 8, 9]])

        fmap = {"Reflectivity()" : data}
        output = feature.calc(None, fmap) 
        self.assertEqual(output.shape, (3,3))
        self.assertAlmostEqual(output[1,1], -1.230000, places=5)
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFeatures)
    unittest.TextTestRunner(verbosity=2).run(suite)
