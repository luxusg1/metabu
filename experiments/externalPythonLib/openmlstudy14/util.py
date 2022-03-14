import os

import lockfile
from sklearn.externals.joblib import load, dump, hash


hashs = dict()


class NumpyMock(object):
    def __init__(self, tmp_dir, array):
        h = hash(array)
        hashs[h] = array
        output_filename = os.path.join(tmp_dir, h)
        with lockfile.LockFile(output_filename):
            if not os.path.exists(output_filename):
                dump(array, output_filename)
        self.filename = output_filename

    def load(self):
        with lockfile.LockFile(self.filename):
            array = load(filename=self.filename)
        return array


class CallbackFunction(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        args = list(args)

        for i in range(len(args)):
            if isinstance(args[i], NumpyMock):
                args[i] = args[i].load()
            elif 'NumpyMock' in str(type(args[i])):
                raise ValueError(type(args[i]))

        args = tuple(args)
        return self.func(*args, **kwargs)