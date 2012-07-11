import unittest2 as unittest
import test_pynexrad.classifier_test

suite = unittest.TestLoader().loadTestsFromModule(test_pynexrad.classifier_test)
unittest.TextTestRunner(verbosity=2).run(suite)
