import unittest2, pyximport, nexrad_util, level2

pyximport.install()

import fast_util

class TestNexradLibrary(unittest2.TestCase):

    def setUp(self):
        self.seq = range(10)

    def test_polar_resample(self):
        sweep = level2.Sweep(r"C:\Users\Reggie\Dev Projects\PyNEXRAD\Test Data\KLIX20050829_133836.Z")
        sweep2 = fast_util.resample_sweep_polar(sweep)
        nexrad_util.display_scan_images([sweep2[:,:,0], sweep2[:,:,1]])


if __name__ == '__main__':
    suite = unittest2.TestLoader().loadTestsFromTestCase(TestNexradLibrary)
    unittest2.TextTestRunner(verbosity=2).run(suite)
