# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  DVN.py
# @Date:  2017/11/16 11:13

import tensorflow as tf
from MyNetLib.NN import conv3D, deconv3D, denseLayer3D
from MyNetLib.Data_Generator import Data
import numpy as np
import os
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

K = 12
MAX_ITER = 15000
Learning_Rate = 0.05
CLASSES = 2
path = '/home/bxsh/airway_data'
BLOCK_SIZE = [32, 64, 64]
stride = [30, 50, 50]

main_stream = []
dense_block1_layers = []
dense_block2_layers = []
DS = None

g = tf.Graph()

DVN_arch = [
    [['block1',

      [  # name,         kernel,          stride,     padding, drop, BN,  ACT,  Norm
          ['conv1', [3, 3, 3, 1, 16], [1, 2, 2, 2, 1], 'SAME', 0.2, False, False, 0.003], ]
      ]
     ],
    [['DenseBlock1',
      [
          ['conv2', [3, 3, 3, 16, K], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv3', [3, 3, 3, 16 + K, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv4', [3, 3, 3, 16 + K * 2, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv5', [3, 3, 3, 16 + K * 3, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv6', [3, 3, 3, 16 + K * 4, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv7', [3, 3, 3, 16 + K * 5, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv8', [3, 3, 3, 16 + K * 6, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv9', [3, 3, 3, 16 + K * 7, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv10', [3, 3, 3, 16 + K * 8, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv11', [3, 3, 3, 16 + K * 9, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv12', [3, 3, 3, 16 + K * 10, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv13', [3, 3, 3, 16 + K * 11, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
      ]
      ]
     ],
    [['block2',
      [

          ['conv14_DS', [1, 1, 1, 16 + K * 12, 160], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['pool', [1, 2, 2, 2, 1], [1, 2, 2, 2, 1], 'SAME'],
      ]
      ]
     ],
    [['DenseBlock2',
      [
          ['conv15', [3, 3, 3, 160, K], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv16', [3, 3, 3, 160 + K, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv17', [3, 3, 3, 160 + K * 2, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv18', [3, 3, 3, 160 + K * 3, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv19', [3, 3, 3, 160 + K * 4, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv20', [3, 3, 3, 160 + K * 5, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv21', [3, 3, 3, 160 + K * 6, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv22', [3, 3, 3, 160 + K * 7, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv23', [3, 3, 3, 160 + K * 8, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv24', [3, 3, 3, 160 + K * 9, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv25', [3, 3, 3, 160 + K * 10, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['conv26', [3, 3, 3, 160 + K * 11, 12], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
      ]
      ]
     ],
    [['block3',
      [
          ['conv27', [1, 1, 1, 160 + K * 12, 304], [1, 1, 1, 1, 1], 'SAME', 0.2, True, True, 0.003],
          ['deconv1', [2, 2, 2, 128, 304], [1, 2, 2, 2, 1], 'SAME'],
          ['deconv2', [2, 2, 2, 64, 128], [1, 2, 2, 2, 1], 'SAME'],
      ]
      ]
     ],
]

with g.as_default():
    with tf.variable_scope('input'):
        X = tf.placeholder(tf.float32, [None, None, None, None, 1])
        Y = tf.placeholder(tf.int64, [None, None, None, None])
        W = tf.placeholder(dtype=tf.float32, shape=[4])
        # rate = tf.placeholder(dtype=tf.float32, shape=[1])
    main_stream.append(X)
    for block in DVN_arch[0]:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = conv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                     layer[6], layer[7])
                        main_stream.append(out)

    dense_block1_layers.append(main_stream[-1])
    for block in DVN_arch[1]:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = denseLayer3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                           layer[6], layer[7])
                        out = tf.concat([dense_block1_layers[-1], out], axis=-1)
                        main_stream.append(out)
                        dense_block1_layers.append(out)

    for block in DVN_arch[2]:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = denseLayer3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                           layer[6], layer[7])
                        main_stream.append(out)
                    if layer[0].startswith('pool'):
                        out = tf.nn.max_pool3d(main_stream[-1], layer[1], layer[2], layer[3])
                        main_stream.append(out)
                    if layer[0].endswith('DS'):
                        DS = main_stream[-1]

    dense_block2_layers.append(main_stream[-1])
    for block in DVN_arch[3]:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = denseLayer3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                           layer[6], layer[7])
                        out = tf.concat([dense_block2_layers[-1], out], axis=-1)
                        main_stream.append(out)
                        dense_block2_layers.append(out)

    for block in DVN_arch[4]:
        with tf.variable_scope(block[0]):
            for layer in block[1]:
                with tf.variable_scope(layer[0]):
                    if layer[0].startswith('conv'):
                        out = denseLayer3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3], layer[4], layer[5],
                                           layer[6], layer[7])
                        main_stream.append(out)
                    if layer[0].startswith('deconv'):
                        out = deconv3D(main_stream[-1], layer[0], layer[1], layer[2], layer[3])
                        main_stream.append(out)

    with tf.variable_scope('DS'):
        with tf.variable_scope('deconv'):
            out = deconv3D(DS, 'DS', [2, 2, 2, 64, 160], [1, 2, 2, 2, 1], "SAME")
        with tf.variable_scope('conv'):
            out = conv3D(out, 'DS', [1, 1, 1, 64, 2], [1, 1, 1, 1, 1], "SAME", None, True, True, 0.003)
        with tf.variable_scope('softmax'):
            inter_out = tf.nn.softmax(out)
            pre_inter = tf.argmax(inter_out, axis=-1)

    with tf.variable_scope('out'):
        with tf.variable_scope('conv'):
            out = conv3D(main_stream[-1], 'DS', [1, 1, 1, 64, 2], [1, 1, 1, 1, 1], "SAME", None, True, True, 0.003)
        with tf.variable_scope('softmax'):
            output = tf.nn.softmax(out)
        with tf.variable_scope('output'):
            pre = tf.argmax(output, axis=-1)

    with tf.variable_scope('LOSS'):
        flat_DS = tf.reshape(inter_out, [-1, CLASSES])
        flat_output = tf.reshape(output, [-1, CLASSES])

        flat_labels = tf.reshape(tf.one_hot(Y, CLASSES), [-1, CLASSES])

        loss_DS = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_DS, labels=flat_labels)

        loss_output = tf.nn.softmax_cross_entropy_with_logits(
            logits=flat_output, labels=flat_labels)

        class_weights = W[0:2]
        weight_map = tf.multiply(flat_labels, class_weights)
        weight_maps = tf.reduce_sum(weight_map, axis=1)

        weighted_loss = tf.multiply(loss_output + loss_DS * W[2], weight_maps)

        loss = tf.reduce_mean(weighted_loss)

    with tf.variable_scope('ACC'):
        ans = tf.equal(pre, Y)
        acc = tf.reduce_mean(tf.cast(ans, tf.float32))

    g_steps = tf.Variable(0)

    train = tf.train.MomentumOptimizer(
        learning_rate=W[3], momentum=0.9).minimize(loss=loss, global_step=g_steps)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged = tf.summary.merge_all()

data = Data(path, BLOCK_SIZE, stride)
if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        x = np.zeros([1, 32, 32, 32, 1])
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        summary_writer = tf.summary.FileWriter('./summary_DVN', sess.graph)
        iteration = 0
        count = 0
        while iteration < MAX_ITER:
            try:
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
                Learning_Rate *= (1.0 - iteration * 1.0 / MAX_ITER) ** 0.9
                w = [portion, 1, 1.0 * (0.99 ** (iteration // 200)),  # [portion,1]
                     Learning_Rate]
                x = 1 - x / 255.0 # 对原始图像取反
                ans1, ans2, ans3 = sess.run(
                    [loss, acc, merged], feed_dict={X: x, Y: y, W: w})
                sess.run(train, feed_dict={X: x, Y: y, W: w})
                if iteration % 100 == 0:
                    count += 1
                    summary_writer.add_summary(ans3, count)

                print(
                    "Iteration:{0},loss:{1},acc:{2},rates:{3},weight:{4}".format(str(iteration),
                                                                                 ans1,
                                                                                 ans2, Learning_Rate, w))
                pic1, pic2 = sess.run([pre, pre_inter], feed_dict={X: x, Y: y, W: w})
                cv2.imwrite(
                    './prediction_DVN/' + str(iteration) + '_raw.jpg',
                    np.uint8(pic1[0, 0, ..., 0] * 255))
                cv2.imwrite(
                    './prediction_DVN/' + str(iteration) + '_pre.jpg',
                    np.uint8(pic1[0, 0, ...]) * 255)
                cv2.imwrite(
                    './prediction_DVN/' + str(iteration) + '_pre_inter.jpg',
                    np.uint8(pic2[0, 0, ...]) * 255)
                cv2.imwrite(
                    './prediction_DVN/' + str(iteration) + '_label.jpg',
                    np.uint8(y[0, 0, ...]) * 255)
            except Exception as e:
                print("出现异常，保存模型")
                print(e)
                saver.save(sess, './test_model_save_DVN/test' + str(iteration) + '.ckpt')

            if iteration % 1000 == 0:
                saver.save(sess, './test_model_save_DVN/test' + str(iteration) + '.ckpt')

            if Learning_Rate - 0 < 0.00001:
                break

        saver.save(sess, './test_model_save_DVN/test.ckpt')
