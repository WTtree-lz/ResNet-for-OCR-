#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
tf 训练识别身份证数字(18个字符)图片

@author: pengyuanjie
"""
import copy
import random
import cv2
import os

import numpy as np
import tensorflow as tf

char_set = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','A']
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 512
IMAGE_CHANNEL = 3


MAX_CAPTCHA = 10
CHAR_SET_LEN = len(char_set)

MAX_STEPS = 100000
Train_data_dir = "D:/program/dataset/milk/gray/dst/train"
Valid_data_dir = "D:/program/dataset/milk/gray/dst/valid"
snapshot_prefix = "D:/program/checkpoints/model.ckpt"
snapshot = 1000
batch_size = 32
valid_step = 1000
max_acc = 0.8

Train_files_list = []
Train_classes = os.listdir(Train_data_dir)
for index, name in enumerate(Train_classes):
    Train_files_list.append(name)

Valid_files_list = []
Valid_classes = os.listdir(Valid_data_dir)
for index, name in enumerate(Valid_classes):
    Valid_files_list.append(name)

# 单字转向量
def char2vec(c):
    vec = np.zeros(len(char_set))
    for j in range(len(char_set)):
        if char_set[j] == c:
            vec[j] = 1
    return vec


def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = cv2.imread(filename)
    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        rgb_image=rgb_image/1.0
    # show_image("src resize image",image)
    return rgb_image


# 生成一个训练batch
def get_next_batch(train = True, batch_size=128, iStart=0, iEnd=128):
    batch_x = np.zeros(shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    if train:
        for i in range(batch_size):
            filename = random.choice(Train_files_list)
            image_path = os.path.join(Train_data_dir, filename)

            image = read_image(image_path, IMAGE_HEIGHT, IMAGE_WIDTH, normalization=True)
            im = image[np.newaxis, :]

            # image_data = cv2.imread(image_path)
            # image_data = cv2.resize(image_data, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # image_data = image_data / 1.0

            ilength = len(filename)
            labels = filename[ilength - 14:ilength - 4]
            vecs = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            for j in range(len(labels)):
                c = labels[j]
                vec = char2vec(c)
                vecs[j * CHAR_SET_LEN:(j + 1) * CHAR_SET_LEN] = np.copy(vec)
            batch_y[i, :] = vecs
            batch_x[i, :] = im
            # batch_x[i, :] = np.reshape(image_data, newshape=(1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
    else:
        filenames = Valid_files_list[iStart:iEnd]
        ilen = len(filenames)
        i=0
        for filename in filenames:
            image_path = os.path.join(Valid_data_dir, filename)

            # image_data = cv2.imread(image_path)
            # image_data = cv2.resize(image_data, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # image_data = image_data / 1.0

            image = read_image(image_path, IMAGE_HEIGHT, IMAGE_WIDTH, normalization=True)
            im = image[np.newaxis, :]

            ilength = len(filename)
            labels = filename[ilength - 14:ilength - 4]
            vecs = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            for j in range(len(labels)):
                c = labels[j]
                vec = char2vec(c)
                vecs[j * CHAR_SET_LEN:(j + 1) * CHAR_SET_LEN] = np.copy(vec)
            batch_y[i, :] = vecs
            batch_x[i, :] = im
            # batch_x[i, :] = np.reshape(image_data, newshape=(1,IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL))
            i = i+1
    return batch_x, batch_y



def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    with tf.variable_scope(scope):
        # beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        # gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, stddev), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def identity_block(X_input, kernel_size, filters, stage, block,is_training):
    """
    Implementation of the identity block as defined in Figure 3

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage"+str(stage)):
        filter1, filter2, filter3 = filters
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,kernel_size=(1, 1), strides=(1, 1),name=conv_name_base+'2a')
        x = batch_norm(x, tf.constant(0.0, shape=[filter1]), tf.random_normal(shape=[filter1], mean=1.0, stddev=0.02), is_training, scope=bn_name_base+'2a')
        # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9, epsilon=0.00001, name=bn_name_base+'2a', training=is_training)
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base+'2b')
        # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9, epsilon=0.00001, name=bn_name_base+'2b', training=is_training)
        x = batch_norm(x, tf.constant(0.0, shape=[filter2]), tf.random_normal(shape=[filter2], mean=1.0, stddev=0.02), is_training, scope=bn_name_base+'2b')
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1),name=conv_name_base+'2c')
        # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9,epsilon=0.00001,name=bn_name_base + '2c', training=is_training)
        x = batch_norm(x, tf.constant(0.0, shape=[filter3]), tf.random_normal(shape=[filter3], mean=1.0, stddev=0.02), is_training, scope=bn_name_base + '2c')
        x = tf.nn.relu(x)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(x, X_shortcut)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result


def convolutional_block(X_input, kernel_size, filters, stage, block, stride, is_training):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):

        # Retrieve Filters
        filter1, filter2, filter3 = filters

        # Save the input value
        X_shortcut = X_input

        # First component of main path
        x = tf.layers.conv2d(X_input, filter1,kernel_size=(1, 1),strides=(stride, stride),name=conv_name_base+'2a')
        # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9, epsilon=0.00001,name=bn_name_base+'2a', training=is_training)
        x = batch_norm(x, tf.constant(0.0, shape=[filter1]), tf.random_normal(shape=[filter1], mean=1.0, stddev=0.02),is_training, scope=bn_name_base+'2a')
        x = tf.nn.relu(x)

        # Second component of main path
        x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b',padding='same')
        # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9, epsilon=0.00001,name=bn_name_base + '2b', training=is_training)
        x = batch_norm(x, tf.constant(0.0, shape=[filter2]), tf.random_normal(shape=[filter2], mean=1.0, stddev=0.02),is_training, scope=bn_name_base + '2b')
        x = tf.nn.relu(x)

        # Third component of main path
        x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
        # x = tf.layers.batch_normalization(x, axis=3,momentum=0.9, epsilon=0.00001, name=bn_name_base + '2c', training=is_training)
        x = batch_norm(x, tf.constant(0.0, shape=[filter3]), tf.random_normal(shape=[filter3], mean=1.0, stddev=0.02),is_training, scope=bn_name_base + '2c')
        x = tf.nn.relu(x)

        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1,1),strides=(stride, stride), name=conv_name_base + '1')
        # X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, momentum=0.9, epsilon=0.00001,name=bn_name_base + '1', training=is_training)
        X_shortcut = batch_norm(X_shortcut, tf.constant(0.0, shape=[filter3]), tf.random_normal(shape=[filter3], mean=1.0, stddev=0.02),is_training, scope=bn_name_base + '1')
        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X_add_shortcut = tf.add(X_shortcut, x)
        add_result = tf.nn.relu(X_add_shortcut)

    return add_result



def ResNet50_reference(X, classes, keep_prob, is_training):

    # x = tf.pad(X, tf.constant([[255, 255], [3, 3], [3, 3], [255, 255]]), "CONSTANT")
    # stage 1
    x = tf.layers.conv2d(X, filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', padding="SAME")
    # x = tf.layers.batch_normalization(x, axis=3, momentum=0.9, epsilon=0.00001,name='bn_conv1')
    x = batch_norm(x, tf.constant(0.0, shape=[64]), tf.random_normal(shape=[64], mean=1.0, stddev=0.02),is_training, scope='bn_conv1')
    x = tf.nn.relu(x)
    x = tf.layers.max_pooling2d(x, pool_size=(3, 3),strides=(2, 2),padding="SAME")
    print(x)
    # stage 2
    x = convolutional_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1,is_training=is_training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',is_training=is_training)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',is_training=is_training)
    print(x)
    # stage 3
    x = convolutional_block(x, kernel_size=3, filters=[128,128,512],stage=3, block='a', stride=2,is_training=is_training)
    x = identity_block(x, 3, [128,128,512], stage=3, block='b',is_training=is_training)
    x = identity_block(x, 3, [128,128,512], stage=3, block='c',is_training=is_training)
    x = identity_block(x, 3, [128,128,512], stage=3, block='d',is_training=is_training)
    print(x)
    # stage 4
    x = convolutional_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', stride=2,is_training=is_training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',is_training=is_training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',is_training=is_training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',is_training=is_training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',is_training=is_training)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',is_training=is_training)
    print(x)
    # stage 5
    x = convolutional_block(x,kernel_size=3,filters=[512, 512, 2048], stage=5, block='a', stride=2,is_training=is_training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',is_training=is_training)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',is_training=is_training)
    print(x)
    # x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(1,1), padding="SAME")
    # print(x)

    w_out = tf.Variable(0.01 * tf.random_normal([2 * 16 * 2048, classes]))
    b_out = tf.Variable(0.1 * tf.random_normal([classes]))
    x = tf.reshape(x, [-1, w_out.get_shape().as_list()[0]])
    print(x)
    out = tf.add(tf.matmul(x, w_out), b_out, name='ocr_out')

    # flatten = tf.layers.flatten(x, name='flatten')
    # out = tf.layers.dense(flatten, units=classes, name='ocr_out')
    return out

# 训练
def train_crack_captcha_cnn():
    X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name='input_data')
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout
    is_training = tf.placeholder(tf.bool, name='is_training')# 是否在训练阶段

    output = ResNet50_reference(X, classes = MAX_CAPTCHA*CHAR_SET_LEN, keep_prob = keep_prob, is_training =is_training)
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=100)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(True, batch_size)
            _, loss_ = sess.run([optimizer, loss],
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8, is_training: True})
            print(step, loss_)
            # 每200 step计算一次准确率
            if step % valid_step == 0 and step != 0:
                epochs = len(Valid_files_list) / batch_size
                accall = []
                accnum = 0
                # for i in range(10):
                for i in range(int(epochs)):
                    iStart = i * batch_size
                    iEnd = (i + 1) * batch_size
                    batch_x_test, batch_y_test = get_next_batch(False, batch_size, iStart, iEnd)
                    acc = sess.run(accuracy,
                                   feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1., is_training: False})
                    num = round(acc * float(batch_size))
                    accnum += num
                    accall.append(acc)
                    print("第%s步训练,第%s步测试***********测试总数：%s  正确总数：%s  验证的准确率为：%s" % (step, i, batch_size, num, acc))
                arr_mean = np.mean(accall)
                print("第%s步训练，训练平均准确率为：%s" % (step, arr_mean))
                accmean = float(accnum) / float(epochs * batch_size)
                print("第%s步训练，-------------- zjs 计算方法 《测试总数：%s，正确总数：%s, 准确率为：%s》" % (
                step, epochs * batch_size, accnum, accmean))
                # 模型保存:每迭代snapshot次或者最后一次保存模型
                if (step % snapshot == 0 and step > 0) or step == MAX_STEPS:
                    print('-----save:{}-{}'.format(snapshot_prefix, step))
                    saver.save(sess, snapshot_prefix, global_step=step)

                # 如果准确率大80%,保存模型,完成训练
                global max_acc
                if arr_mean > max_acc:
                    max_acc = arr_mean
                    print(max_acc)
                    path = os.path.dirname(snapshot_prefix)
                    # save_path = "checkpoints/"
                    best_models = os.path.join(path, 'best_models_{}_{:.8f}.ckpt'.format(step, arr_mean))
                    print('------save:{}'.format(best_models))
                    saver.save(sess, best_models)
                    # break
            step += 1
            if step > MAX_STEPS:
                break
        # writer.close()


if __name__ == '__main__':
    train_crack_captcha_cnn()

















