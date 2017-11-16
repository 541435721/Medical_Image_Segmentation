# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  NN.py
# @Date:  2017/11/16 11:10

import tensorflow as tf


def FC(in_layer, name, out_nodes):
    shape = in_layer.get_shape().as_list()
    nodes = shape[1]
    for i in range(2, len(shape)):
        nodes *= shape[i]
    reshaped = tf.reshape(in_layer, [shape[0], nodes])
    W = tf.get_variable(name + '_W', shape=[nodes, out_nodes], dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(0.0, stddev=0.01))
    b = tf.get_variable(name + '_b', shape=[out_nodes], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    out = tf.matmul(reshaped, W) + b
    return out


def conv2D(in_layer, name, kernel_size, stride, padding='SAME', Dropout=None, BN=False, Activate=False,
           regular=None):
    kernel = tf.get_variable(name=name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.1, stddev=0.01))
    b = tf.get_variable(name=name + '_b', shape=[kernel_size[-1]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('conv2d'):
        out = tf.nn.conv2d(in_layer, kernel, stride, padding)
        out = tf.nn.bias_add(out, b)
    if Dropout:
        with tf.variable_scope('Dropout'):
            out = tf.nn.dropout(out, Dropout)
    if BN:
        with tf.variable_scope('BN'):
            out = tf.layers.batch_normalization(out, training=True)
    if Activate:
        with tf.variable_scope('ACTIVATE'):
            out = tf.nn.relu(out)
    if regular:
        l2_loss = tf.contrib.layers.l2_regularizer(regular)(kernel)
        tf.add_to_collection('regular', l2_loss)
    return out


def deconv2D(in_layer, name, kernel_size, stride, padding='SAME'):
    in_shape = tf.shape(in_layer)
    kernel = tf.get_variable(name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + '_b', shape=[kernel_size[-2]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, kernel_size[-2]])
    with tf.variable_scope('deconv2d'):
        out = tf.nn.conv2d_transpose(in_layer, kernel, output_shape, strides=stride, padding=padding)
        out = tf.nn.relu(tf.nn.bias_add(out, b))
    return out


def conv3D(in_layer, name, kernel_size, stride, padding='SAME', Dropout=None, BN=False, Activate=False,
           regular=None):
    kernel = tf.get_variable(name=name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.1, stddev=0.01))
    b = tf.get_variable(name=name + '_b', shape=[kernel_size[-1]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))

    with tf.variable_scope('conv3d'):
        out = tf.nn.conv3d(in_layer, kernel, stride, padding)
        out = tf.nn.bias_add(out, b)
    if Dropout:
        with tf.variable_scope('Dropout'):
            out = tf.nn.dropout(out, Dropout)
    if BN:
        with tf.variable_scope('BN'):
            out = tf.layers.batch_normalization(out, training=True)
    if Activate:
        with tf.variable_scope('ACTIVATE'):
            out = tf.nn.relu(out)
    if regular:
        l2_loss = tf.contrib.layers.l2_regularizer(regular)(kernel)
        tf.add_to_collection('regular', l2_loss)
    return out


def deconv3D(in_layer, name, kernel_size, stride, padding='SAME'):
    in_shape = tf.shape(in_layer)
    kernel = tf.get_variable(name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.0, 0.01))
    b = tf.get_variable(name + '_b', shape=[kernel_size[-2]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    output_shape = tf.stack(
        [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, kernel_size[-2]])
    with tf.variable_scope('deconv3d'):
        out = tf.nn.conv3d_transpose(in_layer, kernel, output_shape, strides=stride, padding=padding)
        out = tf.nn.relu(tf.nn.bias_add(out, b))
    return out


def denseLayer3D(in_layer, name, kernel_size, stride, padding='SAME', Dropout=None, BN=False, Activate=False,
                 regular=None):
    kernel = tf.get_variable(name=name + '_W', shape=kernel_size, dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(0.1, stddev=0.01))
    b = tf.get_variable(name=name + '_b', shape=[kernel_size[-1]], dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    out = in_layer
    if Dropout:
        with tf.variable_scope('Dropout'):
            out = tf.nn.dropout(out, Dropout)
    if BN:
        with tf.variable_scope('BN'):
            out = tf.layers.batch_normalization(out, training=True)
    if Activate:
        with tf.variable_scope('ACTIVATE'):
            out = tf.nn.relu(out)
    with tf.variable_scope('conv3d'):
        out = tf.nn.conv3d(out, kernel, stride, padding)
        out = tf.nn.bias_add(out, b)
    if regular:
        l2_loss = tf.contrib.layers.l2_regularizer(regular)(kernel)
        tf.add_to_collection('regular', l2_loss)
    return out


if __name__ == '__main__':
    pass
