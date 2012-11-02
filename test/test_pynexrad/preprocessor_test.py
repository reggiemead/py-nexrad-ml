from __future__ import division
import os, sys
import numpy as np

sys.path.append("../")
sys.path.append("../../")
import unittest2 as unittest
import pynexradml.preprocessor as preprocessor
import pynexradml.features as features
import pynexradml.filters as filters

class TestFeature(features.Feature):
    def calc(self, data, fmap):
        return data * 3;

class TestFilter(filters.Filter):
    def apply(self, fmap):
        data = fmap['TestFeature()']
        for i in xrange(len(data)):
            if (data[i] % 2 == 0):
                fmap['_filter_'][i] = 1

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        pass
        
    def testProcessData(self):
        self.processor = preprocessor.Preprocessor(pca=False)
        self.processor.addFeature(TestFeature(), 'TestFeature()')
        self.processor.addFilter(TestFilter(), 'TestFilter()')
        data = np.array([[1, 2, 3], 
                         [4, 5, 6],
                         [7, 8, 9]])
        output = self.processor.processData(data)
        self.assertTrue(np.allclose(output.flatten(), np.array([3, 9, 15, 21, 27])))

    def testPCA(self):
        self.processor = preprocessor.Preprocessor(pca=True)
        self.processor.createAndAddFeature('Reflectivity()')
        self.processor.createAndAddFeature('Velocity()')
        data = np.array([[[2.5, 2.4], [0.5, 0.7]], 
                         [[2.2, 2.9], [1.9, 2.2]],
                         [[3.1, 3.0], [2.3, 2.7]],
                         [[2.0, 1.6], [1.0, 1.1]],
                         [[1.5, 1.6], [1.1, 0.9]]])
        output = self.processor.processData(data)
        self.assertTrue(np.allclose(output[:,0].flatten(), 
            np.array([-.827970186, 1.77758033, -.992197494, -.274210416,
            -1.67580142, -.912949103, .0991094375, 1.14457216, .438046137, 1.22382056]), atol=.00001))
        self.assertTrue(np.allclose(output[:,1].flatten(), 
            np.array([-.175115307, .142857227, .384374989, .130417207,
            -.209498461, .1752824444, -.349824698, .0464172582, .0177646297, -.162675287]), atol=.00001))
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessor)
    unittest.TextTestRunner(verbosity=2).run(suite)
