# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  VoxResNet.py
# @Date:  2017/9/25 20:41


import tensorflow as tf
import numpy as np
import Metrics
import os
import cv2
from Data_Generator import Data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CLASSES = 2
path1 = '/home/bxsh/Data/out.vtk'
path2 = '/home/bxsh/Data/liver'
BLOCK_SIZE = [32, 256, 256]
path = '/home/bxsh/Liver_data'


def conv3D(inlayer, name, kernel, stride, padding, dropout):
    w = tf.get_variable(name + 'w', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-1]], initializer=tf.constant_initializer(0.1))

    out = tf.nn.bias_add(tf.nn.conv3d(inlayer, w, stride, padding), b)
    drop = tf.nn.dropout(out, dropout)
    l2_loss = tf.contrib.layers.l2_regularizer(0.003)(w)
    tf.add_to_collection('l2_loss', l2_loss)
    # BN = tf.layers.batch_normalization(drop, training=True)
    # bias = tf.nn.relu(BN)
    return drop


def deconv3D(inlayer, name, kernel, stride, padding):
    in_shape = tf.shape(inlayer)
    w = tf.get_variable(name + 'W', shape=kernel, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + 'b', shape=[kernel[-2]], dtype=tf.float32, initializer=tf.constant_initializer(0.1))

    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, kernel[-2]])
    deconv = tf.nn.conv3d_transpose(inlayer, w, output_shape, strides=stride, padding=padding)
    bias = tf.nn.relu(tf.nn.bias_add(deconv, b))
    return bias


layers = [
    ['block1', [['conv', [3, 3, 3, 1, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                ['BN_RELU']]],
    ['block2', [['conv', [1, 3, 3, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                ['BN_RELU']]],
    ['block3', [['conv', [3, 3, 3, 32, 64], [1, 2, 2, 2, 1], 'SAME', 0.7],
                ]],
    ['VoxRes1', [['BN_RELU1'],
                 ['conv1', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
    ['VoxRes2', [['BN_RELU1'],
                 ['conv1', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
    ['block4', [['BN_RELU'],
                ['conv1', [3, 3, 3, 64, 64], [1, 2, 2, 2, 1], 'SAME', 0.7]]],
    ['VoxRes3', [['BN_RELU1'],
                 ['conv2', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
    ['VoxRes4', [['BN_RELU1'],
                 ['conv1', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
    ['block5', [['BN_RELU'],
                ['conv', [3, 3, 3, 64, 64], [1, 2, 2, 2, 1], 'SAME', 0.7]]],
    ['VoxRes5', [['BN_RELU1'],
                 ['conv1', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
    ['VoxRes6', [['BN_RELU1'],
                 ['conv1', [1, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['BN_RELU2'],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7]]],
]

main_stream = []
output = []
supervise_layers = [['supervise1', 3, 0, [3, 3, 3, 32, 32], [1, 2, 2, 2, 1]],
                    ['supervise2', 13, 1, [3, 3, 3, 64, 64], [1, 2, 2, 2, 1]],
                    ['supervise3', 23, 2, [3, 3, 3, 64, 64], [1, 2, 2, 2, 1]],
                    ['supervise4', 33, 3, [3, 3, 3, 64, 64], [1, 2, 2, 2, 1]]]
CLASSES = 2

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h,c]
    Y = tf.placeholder(dtype=tf.int64, shape=[None, None, None, None])  # [batch,batchsize,w,h]
    W_class = tf.placeholder(dtype=tf.float32, shape=[CLASSES])
    W_classifier = tf.placeholder(dtype=tf.float32, shape=[5])
    main_stream.append(X)

    for block in layers:
        if block[0].startswith('block'):
            with tf.variable_scope(block[0]):
                for layer in block[1]:
                    if layer[0].startswith('conv'):
                        conv = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4])
                        main_stream.append(conv)
                    if layer[0].startswith('BN'):
                        BN = tf.layers.batch_normalization(main_stream[-1], training=True)
                        bias = tf.nn.relu(BN)
                        main_stream.append(bias)
        if block[0].startswith('VoxRes'):
            with tf.variable_scope(block[0]):
                first_layer = main_stream[-1]
                for layer in block[1]:
                    if layer[0].startswith('conv'):
                        conv = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4])
                        main_stream.append(conv)
                    if layer[0].startswith('BN'):
                        BN = tf.layers.batch_normalization(main_stream[-1], training=True)
                        bias = tf.nn.relu(BN)
                        main_stream.append(bias)
                # main_stream[-1] = tf.concat([first_layer, main_stream[-1]], axis=-1)
                main_stream[-1] = first_layer + main_stream[-1]

    for layer in supervise_layers:
        with tf.variable_scope(layer[0]):
            temp = main_stream[layer[1]]
            for i in range(layer[2]):
                temp = deconv3D(temp, layer[0] + str(i), layer[3], layer[4], 'SAME')
            temp = conv3D(temp, 'conv1_1', [1, 1, 1] + [layer[3][-1]] + [CLASSES], [1, 1, 1, 1, 1], 'SAME', 1.0)
            temp = tf.nn.softmax(temp)
            output.append(temp)

    with tf.variable_scope('predict'):
        out = tf.add_n(output)
        pre = tf.nn.softmax(out)
        output.append(pre)

    with tf.variable_scope('loss'):
        flat_labels = tf.reshape(tf.one_hot(Y, CLASSES), [-1, CLASSES])

        losses = []
        weighted_loss = None
        for i in range(len(output)):
            pre_labels = tf.reshape(output[i], [-1, CLASSES])
            loss_map = tf.nn.softmax_cross_entropy_with_logits(
                logits=pre_labels, labels=flat_labels)

            weight_map = tf.multiply(flat_labels, W_class)
            weight_maps = tf.reduce_sum(weight_map, axis=1)

            weighted_loss = tf.reduce_sum(tf.multiply(loss_map * W_classifier[i], weight_maps))
            losses.append(weighted_loss)
        loss = tf.add_n(losses)

    with tf.variable_scope('trainer'):
        steps = 10000
        g_steps = tf.Variable(0)

        rates = tf.train.exponential_decay(0.2, g_steps, 200, 0.95, staircase=True)
        # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
        train = tf.train.MomentumOptimizer(
            learning_rate=rates, momentum=0.2).minimize(loss=loss, global_step=g_steps)

    with tf.variable_scope('evaluate'):
        pre_img = tf.argmax(pre, -1)
        ans = tf.equal(pre_img, Y)
        acc = tf.reduce_mean(tf.cast(ans, tf.float32))

        # VD = Metrics.VD(pre, Y)
        # VOE = Metrics.VOE(pre, Y)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc', acc)
        merged = tf.summary.merge_all()
        pass

data = Data(path, BLOCK_SIZE)
if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        # saver.restore(sess, './test_model_save_3/test.ckpt')
        tf.global_variables_initializer().run()
        key = 0.0045  # 0.005
        # sess.run(tf.assign(g_steps, 0))
        summary_writer = tf.summary.FileWriter('./summary', graph=sess.graph)
        w = [0.1, 0.2, 0.3, 0.4]
        ans3 = 1000
        # cv2.imwrite('./prediction/test_.jpg', np.uint8(
        # (pic[0, :, :, 0] < pic[0, :, :, 1])) * 255)
        count = 0
        iteration = 0
        while iteration < 100000:
            try:
                try:
                    x, y = data.next()
                except Exception as e:
                    data = Data(path, BLOCK_SIZE)
                    x, y = data.next()

                flat = y.flatten().tolist()

                portion = sum(flat) * 1.0 / (len(flat) - sum(flat))

                if portion < 0.2:
                    continue
                iteration += 1
                w = [portion, 1,
                     1.0 * (0.95 ** (iteration // 200)),  # [portion,1]
                     1.0 * (0.96 ** (iteration // 200)),
                     1.0 * (0.97 ** (iteration // 200)),
                     1.0 * (0.98 ** (iteration // 200)),
                     1.0 * (0.99 ** (iteration // 200))]

                ans1, ans2, ans3, ans4 = sess.run(
                    [loss, acc, rates, merged], feed_dict={X: x, Y: y, W_class: w[:CLASSES], W_classifier: w[CLASSES:]})
                sess.run(train, feed_dict={X: x, Y: y, W_class: w[:CLASSES], W_classifier: w[CLASSES:]})
                if iteration % 100 == 0:
                    count += 1
                    summary_writer.add_summary(ans4, count)

                print(
                    "Iteration:{0},loss:{1},acc:{2},rates:{3},weight:{4}".format(str(iteration),
                                                                                 ans1,
                                                                                 ans2, ans3, w))
                pic = sess.run(pre_img, feed_dict={X: x, Y: y, W_class: w[:CLASSES], W_classifier: w[CLASSES:]})
                cv2.imwrite(
                    './prediction/pre_' + str(iteration) + '.jpg',
                    np.uint8(pic[0, 0, ...]) * 255)
            except Exception as e:
                print("出现异常，保存模型")
                saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

            if iteration % 1000 == 0:
                saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

            if ans3 - 0 < 0.00001:
                break

        saver.save(sess, './test_model_save/test.ckpt')
