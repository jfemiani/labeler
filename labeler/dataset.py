import hashlib
import os
from skimage.io import imsave
import numpy as np
import yaml


def md5(sample):
    name = hashlib.md5(sample.flatten()).hexdigest()
    return name

def touch(fname, times=None):
    os.utime(fname, times)

class Sample(object):
    def __init__(self, dataset, name, data=None, meta={}):
        super(Sample, self).__init__()
        self.dataset = dataset
        self.name = name
        self.data = data
        self.meta = meta

        self.load()
        if data is not None:
            self.data = data
        self.meta.update(meta)

    def save(self):
        dir_name = os.path.dirname(self.name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Save the numpy data
        if self.data is not None:
            np.savez(self.name + '.npz', self.data)

        # Save metadata
        with open(self.name + '.yml', 'w') as f:
            yaml.dump(self.meta, f)

        # Save a thumbnail (shows up in file browsers)
        if self.data is not None and len(self.data.shape) in (2, 3):
            imsave(self.name + '.jpg', self.data)

        self.dataset.set_modified()

    def load(self):
        if os.path.exists(self.name + '.npz'):
            self.data = np.load(self.name  + '.npz')['arr_0']

        if os.path.exists(self.name + '.yml'):
            with open(self.name + '.yml') as f:
                self.meta = yaml.load(f)
        else:
            self.meta = {}


class DataSet(object):
    def __init__(self, name, width, height):
        super(DataSet, self).__init__()
        self.name = name
        self.width = width
        self.height = height
        self.modcount = 0

        #TODO: load modcount from a file

    def set_modified(self):
        self.modcount += 1

        # Sets the modification time for the top-level folder to be the last time the dataset was modified.
        # This way we can quickly check for changes from another process
        touch(self.name)

    def create(self, data, label, meta={}, name=None):
        if name is None:
            name = md5(data)[:10]
        sample_name = '{}/{}x{}/{}/{}'.format(self.name, self.width, self.height, label, name)
        self.set_modified()
        return Sample(self, sample_name, data, meta)

    def delete(self, data, label, name=None):
        if name is None:
            name = md5(data)[:10]
        sample_name = '{}/{}x{}/{}/{}'.format(self.name, self.width, self.height, label, name)

        for ext in '.npz', '.yml', '.jpg':
            if os.path.exists(sample_name + ext):
                os.remove(sample_name + ext)
        self.set_modified()

    def labeled(self, label):
        dir_name = '{}/{}x{}/{}'.format(self.name, self.width, self.height, label)
        for path in os.listdir(dir_name):
            name, ext = os.path.splitext(path)
            if ext == '.yml':
                sample_name = '{}/{}x{}/{}/{}'.format(self.name, self.width, self.height, label, name)
                yield Sample(self, sample_name)
