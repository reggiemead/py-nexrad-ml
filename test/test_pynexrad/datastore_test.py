import os, unittest2, ConfigParser
from datastore import Datastore
from datastore_gui import DatastorePresenter

DATA_DIR = r"F:\Data"

class TestDatastoreFunctions(unittest2.TestCase):

    def setUp(self):
        pass

    def test_manifest(self):
        manifest = ConfigParser.ConfigParser()
        manifest.optionxform = str
        manifest.read(os.path.join(DATA_DIR, "manifest.cfg"))
        for dataset in manifest.sections():
            for (file_name, file_class) in manifest.items(dataset):
                self.assertTrue(os.path.exists(os.path.join(DATA_DIR, file_name)))


"""
    def test_example(self):
        self.assertEqual(len([]), 0)
        self.assertRaises(TypeError, random.shuffle, (1,2,3))
        self.assertTrue(element in self.seq)
"""

if __name__ == '__main__':
    suite = unittest2.TestLoader().loadTestsFromTestCase(TestDatastoreFunctions)
    unittest2.TextTestRunner(verbosity=2).run(suite)
