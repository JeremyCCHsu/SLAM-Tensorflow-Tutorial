# coding=utf8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from readers.fkp_input import load_train_set, BatchRenderer
from models.model1 import SingleHiddenNN

'''
訓練了 400 epoch 後，我們的 validation error 大約在 4e-3
比 Nouri 的數據 (3e-3) 略高。
這個結果其實是很差的! 
Nouri 說 "The predictions look reasonable"，只不過是為了給你信心啊!

但我的實作可能還是有一些問題，提一些給大家參考：
1. 為什麼 Nouri overfits，但我卻沒有?

至於怎麼把這個 model 調好，不是這個 Tutorial 的重點。
你可以把 output layer 換成 tanh
或把座標範圍改到 [0, 1] 函後把 output layer 改成
'''
# [TODO] functions: forward, reverse process, plot

# ======== 這是 tensorflow 吃 program argument 的方式 ========
FLAGS = tf.app.flags.FLAGS  
tf.app.flags.DEFINE_float('lr', 1e-2, 'learning rate')
tf.app.flags.DEFINE_float('valid', 0.2, 'fraction of validation set')
tf.app.flags.DEFINE_integer('n_epoch', 400, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('hidden', 100, 'batch size')

# Global settings
szImg = 96
n_x = szImg * szImg
n_y = 30


def main(args=None):  # pylint: disable=unused-argument
    nn = SingleHiddenNN(
        input_shape=[FLAGS.batch_size, n_x],
        n_hidden=FLAGS.hidden,
        n_y=n_y)
    datasets = load_train_set(valid=FLAGS.valid)
    batches = BatchRenderer(
        datasets.train.images, 
        datasets.train.labels,
        FLAGS.batch_size)
    nn.train(
        batches, 
        datasets.valid, 
        lr=FLAGS.lr, 
        n_epoch=FLAGS.n_epoch)
    display_img_and_prediction(datasets.valid, nn, 4, 
        'imgs/model1-(fully-connected)-face.png')


def display_img_and_prediction(
    valid_set,
    nn,
    n=4, 
    oPngName='imgs/model1-(fully-connected)-face.png'):
    '''
    Plot the results
    '''
    N = n * n
    img = valid_set.images[:N]
    img = np.reshape(img, (N, -1))
    y = valid_set.labels[:N]
    p = nn.predict(img)
    
    plt.figure(figsize=(10, 10))
    for i in range(N):
        plt.subplot(n, n, i+1)
        pn = p[i]
        yn = y[i]
        plt.imshow(img[i].reshape((96, 96)), cmap='gray')
        plt.scatter(yn[0::2] * 48+48, yn[1::2] * 48+48,
        # plt.scatter(yn[0::2] * 96, yn[1::2] * 96,
            marker='o', edgecolors='r', facecolors='none')
        plt.scatter(pn[0::2] * 48+48, pn[1::2] * 48+48,
        # plt.scatter(pn[0::2] * 96, pn[1::2] * 96,
            marker='x', edgecolors='b')
        plt.axis('off')
        plt.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    plt.savefig(oPngName)


if __name__ == '__main__':
    tf.app.run()    # Run啥? 就是 run 本 script 中的 main() (TF根本莫名其妙)
