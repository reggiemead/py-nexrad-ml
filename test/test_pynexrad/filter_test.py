from __future__ import division
import os, sys
import numpy as np

sys.path.append("../")
sys.path.append("../../")
import unittest2 as unittest
import pynexradml.preprocessor as preprocessor
import pynexradml.filters as filters

class TestFilters(unittest.TestCase):
    def setUp(self):
        self.processor = preprocessor.Preprocessor()
        pass
        
    def testDefaultRangeConstraints(self):
        testFilter = self.processor.createFilter("RangeConstraints()")
        self.assertEqual(testFilter.requiredFeatures[0], "Range()")

        fmap = {"Range()" : np.array([19999, 20000, 100000, 145000, 145001]),
                "_filter_" : np.array([0, 0, 0, 0, 0])}
        testFilter.apply(fmap) 
        self.assertTrue(np.allclose(fmap['_filter_'], np.array([1, 0, 0, 0, 1])))

    def testRangeConstraints(self):
        testFilter = self.processor.createFilter("RangeConstraints(10000, 30000)")
        self.assertEqual(testFilter.requiredFeatures[0], "Range()")
        self.assertEqual(testFilter.minRange, 10000)
        self.assertEqual(testFilter.maxRange, 30000)

        fmap = {"Range()" : np.array([9999, 10000, 20000, 30000, 30001]),
                "_filter_" : np.array([0, 0, 0, 0, 0])}
        testFilter.apply(fmap) 
        self.assertTrue(np.allclose(fmap['_filter_'], np.array([1, 0, 0, 0, 1])))
        
    def testRemoveAllBadValues(self):
        testFilter = self.processor.createFilter("RemoveBadValues(Reflectivity)")
        self.assertEqual(testFilter.requiredFeatures[0], "Reflectivity()")

        fmap = {"Reflectivity()" : np.array([filters.BADVAL, 10.0, filters.RFVAL]),
                "_filter_" : np.array([0, 0, 0])}
        testFilter.apply(fmap) 
        self.assertTrue(np.allclose(fmap['_filter_'], np.array([1, 0, 1])))

    def testRemoveBadValues(self):
        testFilter = self.processor.createFilter("RemoveBadValues(Reflectivity, True, False)")
        self.assertEqual(testFilter.requiredFeatures[0], "Reflectivity()")
        self.assertEqual(testFilter.bad, True)
        self.assertEqual(testFilter.rf, False)

        fmap = {"Reflectivity()" : np.array([filters.BADVAL, 10.0, filters.RFVAL]),
                "_filter_" : np.array([0, 0, 0])}
        testFilter.apply(fmap) 
        self.assertTrue(np.allclose(fmap['_filter_'], np.array([1, 0, 0])))

    def testRemoveRangeFoldedValues(self):
        testFilter = self.processor.createFilter("RemoveBadValues(Reflectivity, False, True)")
        self.assertEqual(testFilter.requiredFeatures[0], "Reflectivity()")

        fmap = {"Reflectivity()" : np.array([filters.BADVAL, 10.0, filters.RFVAL]),
                "_filter_" : np.array([0, 0, 0])}
        testFilter.apply(fmap) 
        self.assertTrue(np.allclose(fmap['_filter_'], np.array([0, 0, 1])))

    def testSubSample(self):
        testFilter = self.processor.createFilter("SubSample(.1)")
        fmap = {"_filter_" : np.zeros(100000)}
        testFilter.apply(fmap) 
        self.assertAlmostEqual(sum(fmap["_filter_"]) / 100000, .9, places=2)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFilters)
    unittest.TextTestRunner(verbosity=2).run(suite)
