# coding=utf8

'''
加了 local response normalization 之後會變超慢：
400 epochs 竟然要跑 4:25，真的有夠久 @@ 

沒加的話，大概是 1.57 sec per epoch, 25分鐘可以跑完

另外，由於沒有 regularization，有時候會發生很恐怖的事情
例如 error 降到 0.001 之後突然又飆升到 0.004
然後就回不去了。 
我也不明白到底是為什麼，但我猜是因為 gradient
可能 learning rate 沒有跟著調降所以才這樣的吧?
(Grad explosion?)



'''

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# from util import show_img_keypoint
from util import full_layer, conv_layer, mse, config
# from util import mse

# from model1 import SingleHiddenNN
# def concat(list_of_lists):
#     for 


class CNN(object):
    ''' Convolutional Neural Network (with fully-connected layers) '''
    def __init__(self,
        input_shape=[128, 96, 96, 1], 
        n_filter=[32, 64, 128], 
        n_hidden=[500, 500],
        n_y=30,
        receptive_field=[[3, 3], [2, 2], [2, 2]],
        pool_size=[[2, 2], [2, 2], [2, 2]],
        obj_fcn=mse):
        ''' '''
        self._sanity_check(input_shape, n_filter, receptive_field, pool_size)

        x_shape = input_shape[:]
        x_shape[0] = None
        
        x = tf.placeholder(shape=x_shape, dtype=tf.float32)
        y = tf.placeholder(shape=(None, n_y), dtype=tf.float32)

        self.x, self.y = x, y

        # ========= CNN layers =========
        n_channel = [input_shape[-1]] + n_filter
        for i in range(len(n_channel) -1):
            filter_shape = receptive_field[i] + n_channel[i:i+2] # e.g. [5, 5, 32, 64]
            pool_shape = [1] + pool_size[i] + [1]
            print 'Filter shape (layer %d): %s' % (i, filter_shape)
            conv_and_filter = conv_layer(
                x, filter_shape, 'conv%d' % i, padding='VALID')
            print 'Shape after conv: %s' % conv_and_filter.get_shape().as_list()
            # norm1 = tf.nn.local_response_normalization(
            #     conv_and_filter, 4, bias=1.0, alpha=0.001 / 9.0,
            #     beta=0.75, name='norm%d'%i)
            pool1 = tf.nn.max_pool(
                # norm1,
                conv_and_filter,
                ksize=pool_shape,
                strides=pool_shape, # stride 和 shape 表示不會有重疊
                padding='SAME',
                name='pool%d' % i)
            print 'Shape after pooling: %s' % pool1.get_shape().as_list()
            x = pool1

        # ========= Fully-connected layers =========
        dim = np.prod(x.get_shape()[1:].as_list())
        x = tf.reshape(x, [-1, dim])
        print 'Total dim after CNN: %d' % dim
        for i, n in enumerate(n_hidden):
            x = full_layer(x, n, layer_name='full%d' % i) # nonlinear=tf.nn.relu
        yhat = full_layer(x, n_y, layer_name='output', nonlinear=tf.identity)

        self.yhat = yhat

        self.batch_size = input_shape[0]
        self.lr = tf.placeholder(dtype=tf.float32)

        self.objective = obj_fcn(y, yhat)        
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.objective)
        tf.scalar_summary(self.objective.op.name, self.objective)

        self.sess = tf.Session(config=config)
        

    def predict(self, x):
        ''' Input images (Nx96x96x1), return 30 feature predictors '''
        return self.sess.run(self.yhat, feed_dict={self.x: x})

    def train(self, train_batches, valid_set=None, 
        lr=1e-2, n_epoch=100, logdir='model2'):

        # 呼叫 model saver/loader 物件
        # 實際上是在所有的變數後面加上了 'save' 的操作
        saver = tf.train.Saver(tf.all_variables())

        # 用來記錄例如 training error 等資訊
        summary_op = tf.merge_all_summaries()

        # 叫 graph 把 initialization 排進行程裡
        init = tf.initialize_all_variables()
        self.sess.run(init)  # 做了這步之後，所有的 weight 才會有值

        # graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(
            logdir=logdir,
            graph=self.sess.graph)

        loss_train_record = list() # np.zeros(n_epoch)
        loss_valid_record = list() # np.zeros(n_epoch)
        start_time = time.gmtime()

        n_batch = train_batches.n_batch
        for i in range(n_epoch):
            loss_train_sum = 0.0
            loss_valid_sum = 0.0
            for x, y in train_batches:
                _, loss, summary_str = self.sess.run(
                    [self.optimizer, self.objective, summary_op],
                    feed_dict={
                        self.x: x,
                        self.y: y,
                        self.lr: lr})
                loss_train_sum += loss
                # 事實上，不需要每個epoch 都 validate。因為會拖慢速度。
                if valid_set:
                    loss = self.sess.run(
                        self.objective,
                        feed_dict={
                            self.x: valid_set.images,
                            self.y: valid_set.labels})
                else:
                    loss = 0.0
                loss_valid_sum += loss
            loss_train_sum /= n_batch
            loss_valid_sum /= n_batch

            end_time = time.mktime(time.gmtime())
            # Warning: 這不是 Tensorflow 的 style
            print 'Epoch %04d, %.8f, %.8f,  %0.8f| %.2f sec per epoch' % (
                i, loss_train_sum, loss_valid_sum,
                loss_train_sum/loss_valid_sum,
                (end_time - time.mktime(start_time)) / (i+1))
            loss_train_record.append(loss_train_sum)    # np.log10()
            loss_valid_record.append(loss_valid_sum)    # np.log10()
            if i % 10 == 0:
                ckpt = os.path.join(logdir, 'model.ckpt')
                saver.save(self.sess, ckpt) # 存的是 session
                summary_writer.add_summary(summary_str, i)  # 千萬別太常用，會超慢

        end_time = time.gmtime()
        print time.strftime('%H:%M:%S', start_time)
        print time.strftime('%H:%M:%S', end_time)
        self._error_plot(
            loss_train_record,
            loss_valid_record,
            filename='imgs/model2-(cnn)-loss.png')

    def _sanity_check(self, input_shape, receptive_field, n_filter, pool_size):
        assert len(input_shape) == 4, 'Input size is confined to 2'
        assert len(receptive_field) == len(n_filter), \
            'Inconsistent argument: receptive_field (%d) & n_filter (%d)' % (
                len(receptive_field), len(n_filter))
        assert len(receptive_field) == len(pool_size), \
            'Inconsistent argument: receptive_field (%d) & n_filter (%d)' % (
                len(receptive_field), len(pool_size))

    # def _objective(self, y, yhat):
    #     with tf.variable_scope('Objective'):
    #         obj = tf.sub(y, yhat)   # sub: 矩陣減法。但其實 TF 有operator overloading
    #         obj = tf.square(obj)
    #         # obj = tf.reduce_sum(obj, 1) # 沿著第1軸(橫向)求和
    #         # obj = tf.reduce_mean(obj)
    #         obj = tf.reduce_mean(obj)
    #     return obj

    def _error_plot(self, trn_loss, vld_loss, filename):
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
        plt.savefig(filename)
