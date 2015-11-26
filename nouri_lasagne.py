# file kfkd.py
# This is an excerpt from Daniel Nouri's Tutorial
import os
import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = 'training.csv'
FTEST  = 'test.csv'

def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))


# 
from lasagne import layers
# from lasagne.updates import nesterov_momentum
from lasagne.updates import adagrad
from nolearn.lasagne import NeuralNet

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=adagrad,
    update_learning_rate=0.01,
    # update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=400,  # we want to train this many epochs
    verbose=1,
    )

X, y = load()
net1.fit(X, y)

# train_loss = np.array([i["train_loss"] for i in net1.train_history_])
# valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
# pyplot.plot(train_loss, linewidth=3, label="train")
# pyplot.plot(valid_loss, linewidth=3, label="valid")
# pyplot.grid()
# pyplot.legend()
# pyplot.xlabel("epoch")
# pyplot.ylabel("loss")
# pyplot.ylim(1e-3, 1e-2)
# pyplot.yscale("log")
# pyplot.show()
