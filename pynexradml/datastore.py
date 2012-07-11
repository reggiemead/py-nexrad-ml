import os, sys, tables, level2, ConfigParser
import numpy as np

import pyximport
pyximport.install()

from fast_util import resample_sweep_polar

class ManifestEntry(tables.IsDescription):
    dataset = tables.StringCol(32)
    name = tables.StringCol(32)
    tag = tables.Int32Col()

class Datastore(object):
    def __init__(self, path=""):
        self.path = path

    def rebuild(self):
        ds = tables.openFile(os.path.join(self.path, "datastore.h5"), mode="w", title="Datastore")
        mf_group = ds.createGroup("/", 'manifest', 'Data Manifest')
        mf_table = ds.createTable(mf_group, 'entry', ManifestEntry, "Manifest Entries")
        self.loadDataset(ds, self.path, mf_table)
        ds.close()

    def loadDataset(self, ds, data_path, mf_table):
        manifest = ConfigParser.ConfigParser()
        manifest.optionxform = str
        manifest.read(os.path.join(data_path, "manifest.cfg"))

        for dataset in manifest.sections():
            print "Loading Dataset %s ..." % (dataset)
            if not dataset in ds.root:
                ds_group = ds.createGroup("/", dataset, "%s Dataset" % (dataset))
            for (file_name, file_class) in manifest.items(dataset):
                (node_name, fext) = os.path.splitext(file_name)
                if node_name.startswith("6500"):
                    node_name = node_name[4:]
                print "Loading %s" % (file_name)
                #fill out manifest
                entry = mf_table.row
                entry['dataset'] = dataset
                entry['name'] = node_name
                entry['tag'] = int(file_class)
                entry.append()
                #load data
                data = resample_sweep_polar(level2.Sweep(os.path.join(data_path, file_name)))
                atom = tables.Float32Atom()
                filters = tables.Filters(complib='lzo', complevel=3)
                vscan = ds.createCArray("/" + dataset, node_name, atom, data.shape, filters=filters)
                vscan[:,:,:] = data

        mf_table.flush()
        return manifest

    def importDataset(self, data_path):
        ds = tables.openFile(os.path.join(self.path, "datastore.h5"), mode="a", title="Datastore")

        manifest = ConfigParser.RawConfigParser()
        manifest.optionxform = str
        manifest.read(os.path.join(self.path, "manifest.cfg"))

        imported_manifest = self.loadDataset(ds, data_path, ds.root.manifest.entry)
        for dataset in imported_manifest.sections():
            if not dataset in manifest.sections():
                manifest.add_section(data_set)
            for (file_name, file_class) in imported_manifest.items(dataset):
                manifest.set(dataset, file_name, str(file_class))

        with open(os.path.join(self.path, "manifest.cfg")) as manifestFile:
            manifest.write(manifestFile)

        ds.close()

    def getData(self, dataset, node):
        ds = tables.openFile(os.path.join(self.path, "datastore.h5"), mode="r", title="Datastore")
        result = np.array(ds.getNode('/' + dataset, node))
        ds.close()
        return result

    def getManifest(self, dataset):
        ds = tables.openFile(os.path.join(self.path, "datastore.h5"), mode="r", title="Datastore")
        table = ds.root.manifest.entry
        result = [(x['name'], x['tag']) for x in table.iterrows() if x['dataset'] == dataset]
        ds.close()
        return result

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create and Manage PyNEXRAD Datastores')
    parser.add_argument('-d', '--data_dir')
    parser.add_argument('-r', '--rebuild', action='store_true')
    parser.add_argument('-i', '--import_dir')
    args = parser.parse_args()

    if args.rebuild:
        ds = Datastore(args.data_dir)
        ds.rebuild()
    elif args.import_dir != None:
        ds = Datastore(args.data_dir)
        ds.importDataset(args.import_dir)

