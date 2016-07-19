# coding=utf8

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

szImg = 96

class DataSet(object):   
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s\nlabels.shape: %s" % (
                images.shape, labels.shape))
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        # self._epochs_completed = 0
        # self._index_in_epoch = 0
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_examples(self):
        return self._num_examples
    # # 
    # def next_batch(self, batch_size):
    #     start = self._index_in_epoch
    #     self._index_in_epoch += batch_size
    #     if self._index_in_epoch > self._num_examples:
    #         self._epochs_completed += 1
    #         start = 0
    #         perm  = np.arange(self._num_examples)
    #         np.random.shuffle(perm)
    #         self._images = self._images[perm]
    #         self._labels = self._labels[perm]
    #         self._index_in_epoch = batch_size
    #         assert batch_size <= self._num_examples  # why?
    #     end = self._index_in_epoch
    #     return self._images[start:end], self._labels[start:end]



def str2img(s):
    return np.fromstring(s, sep=' ') / 255.0

def centralize_coordinate(x):
    d = szImg / 2.0
    return (x - d) / d
    # return x / 96

"""
Convert data into Numpy format
"""
def load_train_set(filename='training.csv', valid=0.0, dim=2):
    # Python 中，class 只不過是 dict，只是存取內容的方式改成用 "."
    assert dim == 2 or dim == 4, \
        'Dim can only be 2 (as vectors) or 4 (as w-h-c images)'
    class Datasets(object):
        def __init__(self):
            self.valid = None
            self.train = None

    df = pd.read_csv(filename)
    cols = df.columns[:-1]
    dataset = Datasets()

    df['Image'] = df['Image'].apply(str2img)
    
    # [TODO] Currently dropped, but hopefully they can be reused
    df = df.dropna()  

    X = np.vstack(df['Image'])
    y = centralize_coordinate(df[cols].values)

    if dim == 2:
        pass
    elif dim == 4:
        X = np.reshape(X, (-1, szImg, szImg, 1))

    if valid > 0.0:
        X, y = shuffle(X, y, random_state=42) #  固定 state 是為了讓每次的結果一樣，以便比較
        n = int(valid * len(X))
        dataset.valid = DataSet(X[:n], y[:n])
        dataset.train = DataSet(X[n:], y[n:])
    else:
        dataset.train = DataSet(X, y)
    return dataset


# [TODO]
# def load_test_set(filename='test.csv'):
#     pass


class BatchRenderer(object):
    def __init__(self, X, Y, batch_size, shuffle=True):
        assert X.shape[0] == Y.shape[0], (
            "images.shape: %s\nlabels.shape: %s" % (
                X.shape, Y.shape))
        n_sample = X.shape[0]
        self._n_batch = n_sample // batch_size
        self._batch_index = 0
        self._indices = np.arange(n_sample)
        
        self._X = X
        self._Y = Y
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._n_sample = n_sample
    def __iter__(self):
        return self
    def next(self):
        if self._batch_index >= self._n_batch:
            self._batch_index = 0
            if self._shuffle:
                np.random.shuffle(self._indices)
                # self.X, self.Y = shuffle(self.X, self.Y)
            raise StopIteration
        else:
            i, b = self._batch_index, self._batch_size
            index = self._indices[i*b: (i+1)*b+1]
            self._batch_index += 1
            X = self._X[index]
            Y = self._Y[index]
            return (X, Y)
    @property
    def n_batch(self):
        return self._n_batch

