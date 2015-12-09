# The input image is given in the last field of the data files, 
# and consists of a list of pixels (ordered by row), 
# as integers in (0,255). 
# The images are 96x96 pixels.
# 30 target values (face points)
# 
# 
# 5 min for 400 epochs

# import graphlab
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

# df = graphlab.SFrame('train.csv')

class DataSet(object):
	def __init__(self, images, labels):
		assert images.shape[0] == labels.shape[0], (
			"images.shape: %s\nlabels.shape: %s" % (
				images.shape, labels.shape))
		self._num_examples = images.shape[0]
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0
		# assert images.shape[3] == 1
		# images = images.reshape(
		# 	images.shape[0],
		# 	images.shape[1] * images.shape[2])
	@property
	def images(self):
	  return self._images
	@property
	def labels(self):
	  return self._labels
	@property
	def num_examples(self):
	  return self._num_examples
	@property
	def epochs_completed(self):
	  return self._epochs_completed
	"""
	This method divide the whole sample into batches, 
	and it ignores the final residuals.
	"""
	def next_batch(self, batch_size):
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			self._epochs_completed += 1
			start = 0
			perm  = np.arange(self._num_examples)
			np.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples  # why?
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]


"""
Convert data into Numpy format
"""
def load(test=False, valid=-0.0):
	class Datasets(object):
		pass
	def str2img(s):
		return np.fromstring(s, sep=' ') / 255.0
	def centralize(y):
		y = (y - 48) / 48	# coordinate scaled to [-1, 1]
		return y
	df = pd.read_csv('training.csv')
	cols = df.columns[:-1]
	dataset = Datasets()
	# for col in df.column_names()[:-1]:
	# for col in cols:
	# 	print '%25s: %4d' % (col, df[col].dropna().size())
	df['Image'] = df['Image'].apply(str2img)
	df = df.dropna()		# [TODO] Currently dropped, but hopefully they can be reused
	X  = np.vstack(df['Image']) / 255.0
	if not test:
		y = (df[cols].values -48) / 48.0
		X, y = shuffle(X, y)
	else:
		y = None
	if valid > 0.0:
		n = int(valid * len(X))
		dataset.valid = DataSet(X[:n], y[:n])
		dataset.train = DataSet(X[n:], y[n:])
		# dataset.xvalid = X[:n]
		# dataset.xtrain = X[n:]
		# dataset.yvalid = y[:n]
		# dataset.ytrain = y[n:]
	else:
		dataset.train = DataSet(X, y)
		# dataset.xtrain = X
		# dataset.ytrain = y
	return dataset

"""
  img: a vector of 96*96 length
label: a vector of 30 targets
"""
def show_img_keypoint(img, label, truth=None):
	plt.imshow(img.reshape((96, 96)), cmap='gray')
	plt.scatter(label[0::2] * 48 + 48, label[1::2] * 48 + 48)
	if truth is not None:
		plt.scatter(truth[0::2] * 48 + 48, truth[1::2] * 48 + 48, c='r', marker='x')
	plt.show()


def weight_variable(shape, std=0.0):
	w_initial = tf.truncated_normal(shape, stddev=std)
	return tf.Variable(w_initial)

def bias_variable(shape):
	b_initial = tf.constant(0.0, shape=shape)
	return tf.Variable(b_initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, 
		strides=[1, 1, 1, 1], 
		padding='VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
		strides=[1, 2, 2, 1],
		padding='VALID')


if __name__ == '__main__':
	dataset = load(test=False, valid=0.2)

	iDim = 96*96
	hDim = 100
	oDim = 30
	n_epoch = 400
	BATCH_SIZE = 128

	x  = tf.placeholder("float", shape=[None, iDim])
	y_ = tf.placeholder("float", shape=[None, oDim])

	x  = tf.placeholder("float", shape=[None, iDim])
	y_ = tf.placeholder("float", shape=[None, oDim])

	W  = tf.Variable(
		tf.truncated_normal(
			[iDim, hDim], stddev=1.0/iDim))
	b  = tf.Variable(tf.constant(0.1, shape=[hDim]))
	y1 = tf.nn.relu(tf.matmul(x, W) + b)

	W2 = tf.Variable(
		tf.truncated_normal(
			[hDim, oDim], stddev=1.0/hDim))
	b2 = tf.Variable(tf.constant(0.1, shape=[oDim]))
	y  = tf.matmul(y1, W2) + b2   # tf.nn.softmax(

	loss = tf.reduce_mean(
			tf.reduce_sum(
			tf.square(y - y_), 1))

	train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

	init = tf.initialize_all_variables()
	sess = tf.InteractiveSession()
	sess.run(init)

	loss_train_record = list() # np.zeros(n_epoch)
	loss_valid_record = list() # np.zeros(n_epoch)
	start_time = time.gmtime()
	while dataset.train.epochs_completed < n_epoch:
		batch = dataset.train.next_batch(BATCH_SIZE)
		train_step.run(feed_dict={x: batch[0], y_: batch[1]})
		loss_train = sess.run(loss, feed_dict={x: batch[0], y_: batch[1]})
		loss_valid = sess.run(loss, feed_dict={x: dataset.valid.images, y_: dataset.valid.labels})
		print 'Epoch %04d, %.8f, %.8f,  %0.8f' % (
			dataset.train.epochs_completed, 
			loss_train, loss_valid, 
			loss_train/loss_valid)
		loss_train_record.append(np.log10(loss_train))
		loss_valid_record.append(np.log10(loss_valid))

	end_time = time.gmtime()
	print time.strftime('%H:%M:%S', start_time)
	print time.strftime('%H:%M:%S', end_time)

	i = 0
	img = dataset.valid.images[i]
	lab_y = dataset.valid.labels[i]
	lab_p = sess.run(y, feed_dict={x: dataset.valid.images})[0]
	show_img_keypoint(img, lab_p, lab_y)

	plt.plot(loss_train_record)
	plt.plot(loss_valid_record, c='r')
	plt.show()
