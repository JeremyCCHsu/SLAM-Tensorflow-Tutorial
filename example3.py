# coding=utf8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from readers.fkp_input import load_train_set, BatchRenderer
from models.model3 import CNN

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
tf.app.flags.DEFINE_integer('n_epoch', 1000, 'number of epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
# tf.app.flags.DEFINE_integer('hidden', 100, 'batch size')
tf.app.flags.DEFINE_string('train_dir', 'model_2_cnn', 'dir to store models')

# Global settings
szImg = 96
n_x = szImg * szImg
n_y = 30


def main(args=None):  # pylint: disable=unused-argument
    nn = CNN(
        input_shape=[FLAGS.batch_size, szImg, szImg, 1],
        n_filter=[32, 64, 128],
        n_hidden=[500, 500],
        n_y=30,
        receptive_field=[[3, 3], [2, 2], [2, 2]],
        pool_size=[[2, 2], [2, 2], [2, 2]])
    datasets = load_train_set(valid=FLAGS.valid, dim=4)
    batches = BatchRenderer(
        datasets.train.images,
        datasets.train.labels,
        FLAGS.batch_size)
    nn.train(
        batches,
        datasets.valid,
        lr=FLAGS.lr,
        n_epoch=FLAGS.n_epoch,
        logdir=FLAGS.train_dir)
    display_img_and_prediction(datasets.valid, nn, 4,
        'imgs/model2-(cnn)-face.png')


def display_img_and_prediction(
    valid_set,
    nn,
    n=4,
    oPngName='imgs/model1-(fully-connected)-face.png'):
    '''
    Plot the results.
    (Different from that of fully-connected model)
    '''
    N = n * n
    img = valid_set.images[:N]
    # img = np.reshape(img, (-1, szImg, szImg, 1))
    y = valid_set.labels[:N]
    p = nn.predict(img)

    plt.figure(figsize=(10, 10))
    for i in range(N):
        plt.subplot(n, n, i+1)
        pn, yn = p[i], y[i]
        plt.imshow(img[i][:, :, 0], cmap='gray')
        plt.scatter(yn[0::2] * 48+48, yn[1::2] * 48+48,
            marker='o', edgecolors='r', facecolors='none')
        plt.scatter(pn[0::2] * 48+48, pn[1::2] * 48+48,
            marker='x', edgecolors='b')
        plt.axis('off')
        plt.subplots_adjust(
            left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    
    plt.savefig(oPngName)


if __name__ == '__main__':
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    tf.app.run()    # Run啥? 就是 run 本 script 中的 main() (TF根本莫名其妙)
