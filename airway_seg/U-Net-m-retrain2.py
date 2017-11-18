# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  U-Net-m-retrain2.py
# @Date:  2017/11/17 15:19


import tensorflow as tf
import numpy as np
from .Data_Generator import Data
import cv2
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def conv3D(inlayer, name, kernel, stride, padding='SAME', dropout=0.75):
    w = tf.get_variable(name + 'w', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-1]], initializer=tf.constant_initializer(0.1))

    out = tf.nn.bias_add(tf.nn.conv3d(inlayer, w, stride, padding), b)
    drop = tf.nn.dropout(out, dropout)
    BN = tf.layers.batch_normalization(drop, training=True)
    l2_loss = tf.contrib.layers.l2_regularizer(0.003)(w)
    tf.add_to_collection('l2_loss', l2_loss)
    bias = tf.nn.relu(BN)
    return bias


def deconv3D(inlayer, name, kernel, stride, padding='SAME'):
    in_shape = tf.shape(inlayer)
    w = tf.get_variable(name + 'W', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-2]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, kernel[-2]])
    deconv = tf.nn.conv3d_transpose(inlayer, w, output_shape, strides=stride, padding=padding)
    bias = tf.nn.relu(tf.nn.bias_add(deconv, b))
    return bias


g = tf.Graph()

path = '/home/bxsh/airway_data'
channels = [32, 64, 128, 256, 512]
CLASSES = 2
BLOCK_SIZE = [64, 64, 64]  # 修改了batch_size
stride = [50, 50, 50]

down_architecture = [
    ['block1', [  # 16
        ['conv1', [3, 3, 3, 1, channels[0]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[0], channels[1]], [1, 1, 1, 1, 1]],  # 64
        ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1]]  # 64 shape/2
    ]],
    ['block2', [  # 64
        ['conv1', [3, 3, 3, channels[1], channels[1]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[1], channels[2]], [1, 1, 1, 1, 1]],  # 128
        ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1]]  # 128 shape/2/2
    ]],
    ['block3', [  # 128
        ['conv1', [3, 3, 3, channels[2], channels[2]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[2], channels[3]], [1, 1, 1, 1, 1]],  # 256
        ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1]]  # 256  shape/2/2/2
    ]],
    ['block4', [  # 256
        ['conv1', [3, 3, 3, channels[3], channels[3]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[3], channels[4]], [1, 1, 1, 1, 1]],  # 8*8 shape/2/2/2/2
    ]],
]

up_architecture = [
    ['block5', [  # 512
        ['deconv', [3, 3, 3, channels[4], channels[4]], [1, 2, 2, 2, 1]],
        ['conv1_skip', [3, 3, 3, channels[4] + channels[3], channels[3]], [1, 1, 1, 1, 1]],  # 64*64 shape/2
        ['conv2', [3, 3, 3, channels[3], channels[3]], [1, 1, 1, 1, 1]],  # 64*64 shape/2
    ]],
    ['block6', [  # 256
        ['deconv', [3, 3, 3, channels[3], channels[3]], [1, 2, 2, 2, 1]],
        ['conv1_skip', [3, 3, 3, channels[3] + channels[2], channels[2]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[2], channels[2]], [1, 1, 1, 1, 1]],  # 32*32 shape/2/2
    ]],
    ['block7', [  # 128
        ['deconv', [3, 3, 3, channels[2], channels[2]], [1, 2, 2, 2, 1]],
        ['conv1_skip', [3, 3, 3, channels[2] + channels[1], channels[1]], [1, 1, 1, 1, 1]],
        ['conv2', [3, 3, 3, channels[1], channels[1]], [1, 1, 1, 1, 1]],  # 16*16 shape/2/2/2
        ['conv3', [3, 3, 3, channels[1], 2], [1, 1, 1, 1, 1]],
    ]],
]

main_stream = []
skip_layer = [2, 5, 8]
with g.as_default():
    with tf.variable_scope('data_layer'):
        X = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        Y = tf.placeholder(tf.int64, shape=[None, None, None, None])
        W = tf.placeholder(dtype=tf.float32, shape=[4])
    main_stream.append(X)
    for block in down_architecture:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = conv3D(main_stream[-1], layer[0], layer[1], layer[2])
                        main_stream.append(out)
                    if layer[0].startswith('pool'):
                        out = tf.nn.max_pool3d(main_stream[-1], layer[1], layer[2], padding='SAME')
                        main_stream.append(out)
    for block in up_architecture:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        if layer[0].endswith('skip'):
                            out = tf.concat([main_stream[skip_layer.pop()], main_stream[-1]], axis=-1)
                        else:
                            out = main_stream[-1]
                        out = conv3D(out, layer[0], layer[1], layer[2])
                        main_stream.append(out)
                    if layer[0].startswith('deconv'):
                        out = deconv3D(main_stream[-1], layer[0], layer[1], layer[2])
                        main_stream.append(out)
    with tf.variable_scope('output'):
        out = tf.nn.softmax(main_stream[-1])
        pre = tf.argmax(out, axis=-1)

    with tf.variable_scope('LOSS'):
        with tf.variable_scope('LOSS1'):
            loss1_map = deconv3D(main_stream[5], 'loss1', [3, 3, 3, 2, 128], [1, 2, 2, 2, 1])
            loss1_map = tf.nn.softmax(loss1_map)  # 对中间层softmax输出监督

        with tf.variable_scope('LOSS2'):
            loss2_map = deconv3D(main_stream[8], 'loss2_1', [3, 3, 3, 128, 256], [1, 2, 2, 2, 1])
            loss2_map = deconv3D(loss2_map, 'loss2_2', [3, 3, 3, 2, 128], [1, 2, 2, 2, 1])
            loss2_map = tf.nn.softmax(loss2_map)

        flat_logits = tf.reshape(out, [-1, CLASSES])
        flat_loss1 = tf.reshape(loss1_map, [-1, CLASSES])
        flat_loss2 = tf.reshape(loss2_map, [-1, CLASSES])

        flat_labels = tf.reshape(tf.one_hot(Y, CLASSES), [-1, CLASSES])
        loss_map = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_logits, labels=flat_labels)

        loss1 = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_loss1, labels=flat_labels)

        loss2 = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_loss2, labels=flat_labels)

        class_weights = W[0:2]
        weight_map = tf.multiply(flat_labels, class_weights)
        weight_maps = tf.reduce_sum(weight_map, axis=1)
        weighted_loss = tf.multiply(loss_map + loss1 * W[2] + loss2 * W[3], weight_maps)

        loss = tf.reduce_mean(weighted_loss)

    with tf.variable_scope('ACC'):
        ans = tf.equal(pre, Y)
        acc = tf.reduce_mean(tf.cast(ans, tf.float32))

    steps = 1000
    g_steps = tf.Variable(0)

    rates = tf.train.exponential_decay(0.005, g_steps, 200, 0.95, staircase=True)
    # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
    train = tf.train.MomentumOptimizer(
        learning_rate=rates, momentum=0.2).minimize(loss=loss, global_step=g_steps)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged = tf.summary.merge_all()

data = Data(path, BLOCK_SIZE, stride)
if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './test_model_save_modified_retrain/test.ckpt')
        # tf.global_variables_initializer().run()
        key = 0.0045  # 0.005
        sess.run(tf.assign(g_steps, 0))
        summary_writer = tf.summary.FileWriter('./summary_modified_retrain2', graph=sess.graph)
        w = [0.1, 0.2, 0.3, 0.4]
        ans3 = 1000
        count = 0
        iteration = 0
        while iteration < 15000:
            try:
                try:
                    x, y = data.next()
                except Exception as e:
                    data = Data(path, BLOCK_SIZE, stride)
                    x, y = data.next()

                flat = y.flatten().tolist()

                portion = sum(flat) * 1.0 / (len(flat) - sum(flat))

                if portion < 0.01:
                    continue
                iteration += 1
                w = [1, 1, 0.03 * (0.8 ** (iteration // 200)),  # [portion,1]
                     0.04 * (0.8 ** (iteration // 200))]

                ans1, ans2, ans3, ans4 = sess.run(
                    [loss, acc, rates, merged], feed_dict={X: x, Y: y, W: w})
                sess.run(train, feed_dict={X: x, Y: y, W: w})
                if iteration % 100 == 0:
                    count += 1
                    summary_writer.add_summary(ans4, count)

                print(
                    "Iteration:{0},loss:{1},acc:{2},rates:{3},weight:{4}".format(str(iteration),
                                                                                 ans1,
                                                                                 ans2, ans3, w))
                pic = sess.run(pre, feed_dict={X: x, Y: y, W: w})
                cv2.imwrite(
                    './prediction_modified_retrain2/pre_' + str(iteration) + '.jpg',
                    np.uint8(pic[0, 0, ...]) * 255)
            except Exception as e:
                print("出现异常，保存模型")
                saver.save(sess, './test_model_save_modified_retrain2/test' + str(iteration) + '.ckpt')

            if iteration % 1000 == 0:
                saver.save(sess, './test_model_save_modified_retrain2/test' + str(iteration) + '.ckpt')

            if ans3 - 0 < 0.00001:
                break

        saver.save(sess, './test_model_save_modified_retrain2/test.ckpt')
