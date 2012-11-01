import sys
import unittest2 as unittest
import test_pynexrad.filter_test
import test_pynexrad.feature_test

suite = unittest.TestSuite()
suite.addTests(unittest.TestLoader().loadTestsFromModule(test_pynexrad.feature_test))
suite.addTests(unittest.TestLoader().loadTestsFromModule(test_pynexrad.filter_test))
unittest.TextTestRunner(verbosity=2).run(suite)
