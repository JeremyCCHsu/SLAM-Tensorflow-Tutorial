# coding=utf8
'''
Computational graph programming 和一般寫程式的不同
需要先把整個資料與運算的流程定義好
然後再把 data 像水一樣傾倒進 computational graph

一般在寫程式的時候，一行程式代表一個運算 (operation)
所以如果像是 Python 這種直譯式語言 (可以一行一行執行的這種)
程式寫到哪，資料和運算就執行到哪

但 Tensorflow 不是
你是先把 Computational graph 建好
再把 data 灌進去。

如果把編寫程式比喻成骨牌
寫 Python 的時候，你可以在每個時間點把時間凍結住，看現在發生什麼事
但寫 Tensorflow 的時候卻不能。
你只能把骨牌排好。在執行的時候，你才知道哪邊出問題。
'''

'''
關於 Tensorflow 的變數

兩種宣告的方法：
1. tf.Variable(initial_value, name, dtype): 最簡單的方法
2. tf.get_variable(name, initializer): 比較麻煩，但在 variable sharing 石只能這樣用




「變數」的種類 (不要先介紹，會讓人發瘋)

1. tf.Variable: 基本上就是 NN 的 parameters (weight, bias) 
2. tf.placeholder: 基本上就是 input
    宣告這種變數等於是告訴 graph 說：「我等一下才要把 data 餵進來」
2. Operation (op) 的結果: 如 tf.add, tf.matmul。
    雖說是 op output，但實際上這些中間產物常常就是甚至是數學式子上的變數,
    e.g.  
        x = tf.Variable(tf.zeros([1]))
        y = tf.add(x, 1)
3. Random variable: 每次存取的時候值都會變
    *注意: 通常 weight 在宣告的時候，會用 random variable 當作「初始值」
           由於只是當初使值，後來並不會再被 RNG 更動了
           不要因為寫
           w = tf.Variable(tf.random_normal([128, 2], 0, 1))
           就被搞糊塗了
3. Constants: 通常也是拿來初始的時候才會用到。Constant 是不會被更動的。





'''
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from util import show_img_keypoint
from util import full_layer, config

class IModel(object):
    def train(self, x):
        pass
    def predict(self, x):
        pass


class SingleHiddenNN(IModel):
    def __init__(self, input_shape=[128, 9216], n_hidden=100, n_y=30):
        '''
        input_shape: (batch_size, input dimension), tuple
        n_hidden: # latent variables
        n_y: # output features.
        '''
        batch_size, n_x = input_shape
        x = tf.placeholder(shape=(None, n_x), dtype=tf.float32, name='x')
        y = tf.placeholder(shape=(None, n_y), dtype=tf.float32, name='y')

        h1 = full_layer(x, n_hidden, 'Hidden01')
        yhat = full_layer(h1, n_y, 'Output', nonlinear=tf.identity)

        self.batch_size = batch_size
        self.lr = tf.placeholder(dtype=tf.float32)
        self.x = x
        self.y = y
        self.yhat = yhat

        self.obj_fcn = self._objective(y, yhat)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.obj_fcn)

        self.sess = tf.Session(config=config)
        

    def predict(self, x):
        return self.sess.run(self.yhat, feed_dict={self.x: x})

    def train(self, train_batches, valid_set=None, lr=1e-2, n_epoch=100):
        # [TODO] Need to deal with empty validation
        
        # 叫 graph 把 initialization 排進行程裡
        init = tf.initialize_all_variables() 
        self.sess.run(init)  # 做了這步之後，所有的 weight 才會有值

        # graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(
            logdir='model1',
            graph=self.sess.graph)

        loss_train_record = list() # np.zeros(n_epoch)
        loss_valid_record = list() # np.zeros(n_epoch)
        start_time = time.gmtime()

        n_batch = train_batches.n_batch
        for i in range(n_epoch):
            loss_train_sum = 0.0
            loss_valid_sum = 0.0
            for x, y in train_batches:
                _, loss_train = self.sess.run(
                    [self.optimizer, self.obj_fcn], 
                    feed_dict={
                        self.x: x, 
                        self.y: y,
                        self.lr: lr})
                loss_valid = self.sess.run(
                    self.obj_fcn, 
                    feed_dict={
                        self.x: valid_set.images, 
                        self.y: valid_set.labels})
                loss_train_sum += loss_train
                loss_valid_sum += loss_valid
            # Warning: 這不是 Tensorflow 的 style
            print 'Epoch %04d, %.8f, %.8f,  %0.8f' % (
                i, loss_train_sum/n_batch, loss_valid_sum/n_batch,
                loss_train_sum/loss_valid_sum)
            loss_train_record.append(loss_train_sum)    # np.log10()
            loss_valid_record.append(loss_valid_sum)    # np.log10()



        end_time = time.gmtime()
        print time.strftime('%H:%M:%S', start_time)
        print time.strftime('%H:%M:%S', end_time)
        self._error_plot(loss_train_record, loss_valid_record)

    def _objective(self, y, yhat):
        '''
        I initially sum up the row and compute the mean.
        It ended up error = 0.12
        When comparing to Nouri's error= 0.003, mine sucked.
        However, hs (and Kaggle alike) took mean over the row.
        No wonder his results were way better than mine! = =
        '''
        with tf.variable_scope('Objective'):
            obj = tf.sub(y, yhat)   # sub: 矩陣減法。但其實 TF 有operator overloading
            obj = tf.square(obj)
            # obj = tf.reduce_sum(obj, 1) # 沿著第1軸(橫向)求和
            # obj = tf.reduce_mean(obj)
            obj = tf.reduce_mean(obj)
        return obj

    def _error_plot(self, trn_loss, vld_loss):
        '''
        trn_loss: training_loss_record
        vld_loss: validation_loss_record
        '''
        plt.figure()
        plt.plot(trn_loss, label='train')
        plt.plot(vld_loss, c='r', label='validation')
        plt.xlabel('mini-batch')
        plt.ylabel('loss')
        plt.yscale('log')
        # plt.ylim(1e-3, 1e-2)
        plt.legend()
        plt.savefig('imgs/model1-(fully-connected)-loss.png')
