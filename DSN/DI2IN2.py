# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  DI2IN.py
# @Date:  2017/10/23 14:19



import tensorflow as tf
import numpy as np
import cv2
import SimpleITK as sitk
import os
from Data_Generator import Data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CLASSES = 2
BLOCK_SIZE = [32, 256, 256]
stride = [6, 128, 128]
path = '/home/bxsh/Liver_data'

layers = [['block1', [['conv1', [3, 3, 3, 1, 16], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['conv2', [3, 3, 3, 16, 16], [1, 1, 2, 2, 1], 'SAME', 0.7], ]],
          ['block2', [['conv1', [3, 3, 3, 16, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['conv2', [3, 3, 3, 32, 32], [1, 1, 2, 2, 1], 'SAME', 0.7], ]],
          ['block3', [['conv1', [3, 3, 3, 32, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['conv2', [3, 3, 3, 64, 64], [1, 1, 2, 2, 1], 'SAME', 0.7], ]],
          ['block4', [['conv1', [3, 3, 3, 64, 128], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['conv2', [3, 3, 3, 128, 128], [1, 1, 2, 2, 1], 'SAME', 0.7], ]],
          ['block5', [['conv1', [3, 3, 3, 128, 256], [1, 1, 1, 1, 1], 'SAME', 0.7],
                      ['conv2', [3, 3, 3, 256, 256], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
          ['block6', [['up', 2, [8, 10]],
                      ['conv', [3, 3, 3, 384, 128], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
          ['block7', [['up', 2, [6, 12]],
                      ['conv', [3, 3, 3, 192, 64], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
          ['block8', [['up', 2, [4, 14]],
                      ['conv', [3, 3, 3, 96, 32], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
          ['block9', [['up', 2, [2, 16]],
                      ['conv', [3, 3, 3, 48, 16], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
          ]

supervise_block = [
    ['block10', [['up', 16, 10],
                 ['conv1', [3, 3, 3, 256, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv2', [3, 3, 3, 8, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv3', [3, 3, 3, 8, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv4', [3, 3, 3, 8, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv5', [3, 3, 3, 8, 1], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
    ['block11', [['up', 4, 14],
                 ['conv1', [3, 3, 3, 64, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv2', [3, 3, 3, 8, 8], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv3', [3, 3, 3, 8, 1], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
    ['block12', [['up', 1, 18],
                 ['conv', [3, 3, 3, 16, 1], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
    ['block13', [['conv', [3, 3, 3, 3, 1], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
]
discriminator_architecture = [
    ['block14', [['conv1', [3, 3, 3, 1, 16], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv2', [3, 3, 3, 16, 16], [1, 2, 2, 2, 1], 'SAME', 0.7], ]],
    ['block15', [['conv1', [3, 3, 3, 16, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv2', [3, 3, 3, 32, 32], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv3', [3, 3, 3, 32, 32], [1, 2, 2, 2, 1], 'SAME', 0.7], ]],
    ['block16', [['conv1', [3, 3, 3, 32, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv2', [3, 3, 3, 64, 64], [1, 1, 1, 1, 1], 'SAME', 0.7],
                 ['conv3', [3, 3, 3, 64, 1], [1, 1, 1, 1, 1], 'SAME', 0.7], ]],
]

supervise_stream = []
concat_layers = [6, 7, 8, 9]
out = []
surprise_layers = [5, 7, 9]


def conv3D(inlayer, name, kernel, stride, padding, dropout):
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


def upscale(inlayer, factor):
    temp = tf.squeeze(inlayer, axis=0)
    in_shape = tf.shape(temp)
    up = tf.image.resize_images(temp,
                                tf.stack([in_shape[1] * factor, in_shape[2] * factor]))
    return tf.expand_dims(up, 0)


def generator(X, reuse=False):
    main_stream = []
    main_stream.append(X)
    with tf.variable_scope('generator', reuse=reuse):
        for block in layers:
            with tf.variable_scope(block[0], reuse=reuse):
                for layer in block[1]:
                    with tf.variable_scope(layer[0], reuse=reuse):
                        if layer[0].startswith('up'):
                            bridge = tf.concat([main_stream[layer[-1][0]], main_stream[layer[-1][1]]], axis=-1)
                            up = upscale(bridge, layer[1])
                            main_stream.append(up)
                        if layer[0].startswith('conv'):
                            conv = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4])
                            main_stream.append(conv)
        count = 0
        new_layer = None
        for block in supervise_block:
            if count >= 3:
                with tf.variable_scope(block[0], reuse=reuse):
                    new_layer = tf.concat(supervise_stream, axis=-1)
                    for layer in block[1]:
                        if layer[0].startswith('conv'):
                            new_layer = conv3D(new_layer, layer[0], layer[1], layer[2], layer[3], layer[4])
                            main_stream.append(new_layer)
                break
            with tf.variable_scope(block[0], reuse=reuse):
                for layer in block[1]:
                    with tf.variable_scope(layer[0], reuse=reuse):
                        if layer[0].startswith('up'):
                            new_layer = main_stream[layer[2]]
                            new_layer = upscale(new_layer, layer[1])
                        if layer[0].startswith('conv'):
                            new_layer = conv3D(new_layer, layer[0], layer[1], layer[2], layer[3], layer[4])
                supervise_stream.append(new_layer)
                count += 1
    return tf.sigmoid(main_stream[-1])


def discriminator(X, reuse=False):
    main_stream = []
    main_stream.append(X)
    with tf.variable_scope('discriminator', reuse=reuse):
        for block in discriminator_architecture:
            with tf.variable_scope(block[0], reuse=reuse):
                for layer in block[1]:
                    with tf.variable_scope(layer[0], reuse=reuse):
                        conv = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4])
                        main_stream.append(conv)
    hidden = tf.reshape(main_stream[-1], [-1])
    with tf.variable_scope('FC', reuse=reuse):
        w = tf.get_variable('FCw', shape=[32768], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(0, 0.01))
        hidden = tf.reduce_sum(tf.multiply(w, hidden))
        main_stream.append(tf.sigmoid(hidden))  #
    out.append(main_stream[-1])
    return main_stream[-1]


g = tf.Graph()
with g.as_default():
    with tf.variable_scope('input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h,c]
        Y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h]
        W = tf.placeholder(dtype=tf.float32, shape=[4])

    pre = generator(X)
    p1 = discriminator(Y)
    p2 = discriminator(pre, True)

    weight1 = (tf.reshape(Y, [-1]))
    weight2 = (1 - weight1)
    weight = tf.add(weight1 * W[0], weight2 * W[1])
    flatten_pre = tf.reshape(pre, [-1])
    weighted_pre = tf.multiply(weight, flatten_pre)
    flatten_Y = tf.reshape(Y, [-1])

    acc = 1 - tf.reduce_mean(tf.abs((tf.cast(flatten_pre > 0.9, tf.float32) - flatten_Y)))

    loss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=flatten_Y,
                                                logits=weighted_pre)) + tf.nn.sigmoid_cross_entropy_with_logits(
        labels=1.0, logits=p2)
    loss_d = tf.nn.sigmoid_cross_entropy_with_logits(labels=1.0, logits=p1) + tf.nn.sigmoid_cross_entropy_with_logits(
        labels=0.0, logits=p2)

    train_vars = tf.trainable_variables()
    g_vars = [var for var in train_vars if var.name.startswith('generator')]
    d_vars = [var for var in train_vars if var.name.startswith('discriminator')]

    steps = 1000
    g_steps = tf.Variable(0)

    rates = tf.train.exponential_decay(0.2, g_steps, 200, 0.95, staircase=True)
    # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
    train1 = tf.train.MomentumOptimizer(
        learning_rate=rates, momentum=0.2).minimize(loss=loss_g, global_step=g_steps)

    train2 = tf.train.MomentumOptimizer(
        learning_rate=rates, momentum=0.2).minimize(loss=loss_d, global_step=g_steps, var_list=d_vars)

    tf.summary.scalar('loss_d', loss_d)
    tf.summary.scalar('loss_g', loss_g)
    merged = tf.summary.merge_all()

data = Data(path, BLOCK_SIZE, stride)
if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './test_model_save4000/test.ckpt')
        key = 0.0045  # 0.005
        summary_writer = tf.summary.FileWriter('./summary_1', graph=sess.graph)

        w = [1, 2, 3, 4]
        count = 0
        iteration = 0
        while iteration < 10000:
            # try:
            try:
                x, y = data.next()
            except Exception as e:
                data = Data(path, BLOCK_SIZE, stride)
                x, y = data.next()

            flat = y.flatten().tolist()

            portion = sum(flat) * 1.0 / (len(flat) - sum(flat))

            if portion < 0.1:
                continue
            iteration += 1

            w = [portion, 1, 0.8 * (0.99 ** (iteration // 200)),
                 1.0 * (0.99 ** (iteration // 200))]
            y = y[..., np.newaxis]
            ans1, ans2, ans3, ans4, ans5 = sess.run(
                [loss_g, loss_d, rates, acc, merged], feed_dict={X: x, Y: y, W: w})
            sess.run(train1, feed_dict={X: x, Y: y, W: w})
            sess.run(train2, feed_dict={X: x, Y: y, W: w})
            if iteration % 100 == 0:
                count += 1
                summary_writer.add_summary(ans5, count)

            print(
                "Iteration:{0},loss_g:{1},loss_d:{2},acc:{3},rates:{4},weight:{5}".format(str(iteration),
                                                                                          ans1, ans2,
                                                                                          ans4, ans3, w[0:2]))
            pic = sess.run(pre, feed_dict={X: x, Y: y, W: w})
            cv2.imwrite(
                './prediction/pre_' + str(iteration) + '.jpg',
                np.uint8(pic[0, 0, :, :, 0] > 0.9) * 255)

            # except Exception as e:
            #     print("出现异常，保存模型")
            #     saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

            if iteration % 1000 == 0:
                saver.save(sess, './test_model_save/test' + str(iteration) + '.ckpt')

                # if ans3 - 0 < 0.00001:
                #     break

        saver.save(sess, './test_model_save/test.ckpt')
