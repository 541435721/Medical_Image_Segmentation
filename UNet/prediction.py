# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  prediction.py
# @Date:  2017/10/18 10:27


import tensorflow as tf
import numpy as np
import cv2
import SimpleITK as sitk
import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

CLASSES = 2
path1 = '/home/bxsh/Data/out.vtk'
path2 = '/home/bxsh/Data/liver'
BLOCK_SIZE = [8, 500, 500]
stride = [6, 128, 128]
path = '/home/bxsh/Liver_data_norm'


def conv3d(name, in_layer, ksize, out_channels, padding='SAME', in_channel=0):
    with tf.variable_scope(name):
        if in_channel:
            W = tf.get_variable(name + 'W', shape=[ksize[0], ksize[1], ksize[2], in_channel, out_channels],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(0.0, 0.01))
        else:
            W = tf.get_variable(name + 'W',
                                shape=[ksize[0], ksize[1], ksize[2], in_layer.get_shape()[-1], out_channels],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(0.0, 0.01))
        b = tf.get_variable(
            name + 'b', shape=[out_channels], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
    with tf.name_scope(name + "op"):
        conv = tf.nn.conv3d(in_layer, filter=W, strides=[
            1, 1, 1, 1, 1], padding=padding)
        drop = tf.nn.dropout(conv, 0.75)
        temp = tf.nn.bias_add(drop, b)
        BN = tf.layers.batch_normalization(temp, training=True)
        bias = tf.nn.relu(BN)
        l2_loss = tf.contrib.layers.l2_regularizer(0.003)(W)
        tf.add_to_collection('l2_loss', l2_loss)
    return bias


def deconv3d(name, in_layer, ksize, out_channels, padding='SAME'):
    with tf.variable_scope(name):
        in_shape = tf.shape(in_layer)
        W = tf.get_variable(name + 'W', shape=[ksize[0], ksize[1], ksize[2], out_channels, in_layer.get_shape()[-1]],
                            dtype=tf.float32, initializer=tf.truncated_normal_initializer(0.0, 0.01))
        b = tf.get_variable(name + 'b', shape=[out_channels], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
    with tf.name_scope(name + 'op'):
        output_shape = tf.stack(
            [in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] * 2, out_channels])
        deconv = tf.nn.conv3d_transpose(in_layer, W, output_shape, strides=[
            1, 2, 2, 2, 1], padding=padding)
        bias = tf.nn.relu(tf.nn.bias_add(deconv, b))
    return bias


g = tf.Graph()

with g.as_default():
    with tf.name_scope('input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None, 1])  # [batch,batchsize,w,h,c]
        Y = tf.placeholder(dtype=tf.int64, shape=[None, None, None, None])  # [batch,batchsize,w,h]
        W = tf.placeholder(dtype=tf.float32, shape=[4])

    root_layers = 32
    with tf.name_scope('L1'):
        conv1_1 = conv3d('conv1_1', X, [3, 3, 3],
                         root_layers, padding="SAME")  # batch_size*512*512*64
        conv1_2 = conv3d('conv1_2', conv1_1, [
            3, 3, 3], root_layers, padding='SAME')  # 融合层1 512*512*64

    with tf.name_scope('L2'):
        pool1 = tf.nn.max_pool3d(conv1_2, ksize=[1, 2, 2, 2, 1], strides=[  # 256*256*64
            1, 2, 2, 2, 1], padding='SAME')

        conv2_1 = conv3d('conv2_1', pool1, [
            3, 3, 3], root_layers * 2, padding="SAME")  # 256*256*128
        conv2_2 = conv3d('conv2_2', conv2_1, [
            3, 3, 3], root_layers * 2, padding="SAME")  # 融合层2 256*256*128

    with tf.name_scope('L3'):
        pool2 = tf.nn.max_pool3d(conv2_2, ksize=[1, 2, 2, 2, 1], strides=[  # 128*128*128
            1, 2, 2, 2, 1], padding="SAME")

        conv3 = conv3d('conv3', pool2, [
            3, 3, 3], root_layers * 4, padding="SAME")  # 128*128*256
        conv4 = conv3d('conv4', conv3, [
            3, 3, 3], root_layers * 4, padding="SAME")  # 128*128*256

    with tf.name_scope('DL1'):
        deconv1 = deconv3d('deconv1', conv4, [
            3, 3, 3], root_layers * 2, padding='SAME')  # 128*128*128

        fusing1 = tf.concat([conv2_2, deconv1], axis=-1)  # 融合层1 256*256*256

        conv5_1 = conv3d('conv5_1', fusing1, [
            3, 3, 3], root_layers * 2, in_channel=root_layers * 4, padding="SAME")  # 256*256*128
        conv5_2 = conv3d('conv5_2', conv5_1, [
            3, 3, 3], root_layers * 2, padding="SAME")  # 256*256*128

    with tf.name_scope('LOSS1'):
        loss1_map = conv3d('loss1_map', conv1_2, [3, 3, 3], CLASSES, padding="SAME")

    with tf.name_scope('LOSS2'):
        loss2_map = deconv3d('loss2_map', conv2_2, [3, 3, 3], CLASSES, padding="SAME")

    with tf.name_scope('DL2'):
        deconv2 = deconv3d('deconv2', conv5_2, [
            3, 3, 3], root_layers, padding='SAME')  # 256*256*64

        fusing2 = tf.concat([conv1_2, deconv2], -1)  # 融合层2 512*512*128

        conv6_1 = conv3d('conv6_1', fusing2, [
            3, 3, 3], root_layers, in_channel=root_layers * 2, padding="SAME")
        conv6_2 = conv3d('conv6_2', conv6_1, [
            3, 3, 3], root_layers, in_channel=root_layers, padding="SAME")

    with tf.name_scope('out'):
        conv7 = conv3d('conv7', conv6_2, [
            1, 1, 1], CLASSES, in_channel=root_layers, padding="SAME")  # 512*512*2
        pre = tf.nn.softmax(conv7)

    flat_logits = tf.reshape(pre, [-1, CLASSES])
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

    loss = tf.reduce_mean(weighted_loss) + tf.add_n(tf.get_collection('l2_loss'))

    pre_img = tf.argmax(pre, -1)
    ans = tf.equal(pre_img, Y)
    acc = tf.reduce_mean(tf.cast(ans, tf.float32))

    steps = 1000
    g_steps = tf.Variable(0)

    rates = tf.train.exponential_decay(0.0001, g_steps, 200, 0.95, staircase=True)
    # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
    train = tf.train.MomentumOptimizer(
        learning_rate=rates, momentum=0.2).minimize(loss=loss, global_step=g_steps)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged = tf.summary.merge_all()


def read_dcm(names, raw=False):
    if raw:
        img = sitk.ReadImage(names)
    else:
        names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(names)
        img = sitk.ReadImage(names)
    return sitk.GetArrayFromImage(img)


def segmentation(vtk_file, stride, method=1):
    start = time.clock()
    with tf.Session(graph=g) as sess:
        restore_model = time.clock()
        saver = tf.train.Saver()
        saver.restore(sess, './test_model_save_5/test.ckpt')#test_model_save_norm4
        print('restore Model:', time.clock() - restore_model)
        img = sitk.ReadImage(vtk_file)
        img_arr = sitk.GetArrayFromImage(img)
        input_arr = img_arr[np.newaxis, ..., np.newaxis]
        out_arr1 = np.zeros_like(img_arr)[np.newaxis]
        out_arr2 = np.zeros_like(img_arr)[np.newaxis]
        out_arr3 = np.zeros_like(img_arr)[np.newaxis]
        seg_start = time.clock()
        for z in range(0, img_arr.shape[0] - stride, stride):
            patch = input_arr[:, z:z + stride, ...]
            pre_mask = sess.run(pre_img, feed_dict={X: patch})
            out_arr1[:, z:z + stride, ...] = np.uint8(pre_mask)

        for z in range(0, img_arr.shape[0] - stride * 2, stride * 2):
            patch = input_arr[:, z:z + stride * 2, ...]
            pre_mask = sess.run(pre_img, feed_dict={X: patch})
            out_arr2[:, z:z + stride * 2, ...] = np.uint8(pre_mask)
        for z in range(0, img_arr.shape[0] - stride * 4, stride * 4):
            patch = input_arr[:, z:z + stride * 4, ...]
            pre_mask = sess.run(pre_img, feed_dict={X: patch})
            out_arr3[:, z:z + stride * 4, ...] = np.uint8(pre_mask)
        out_img = np.squeeze((out_arr1 + out_arr2 + out_arr3) // 2)
        print('run web:', time.clock() - seg_start)
        index = 0
        maxN = 0
        fineturn_start = time.clock()
        for z in range(out_img.shape[0]):
            S = out_img[z, :, :].sum()
            if S > maxN:
                maxN = S
                index = z
        print(index, maxN)
        print(out_img[index, 125, 229])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for z in range(index, 0, -1):
            out_img[z - 1] *= cv2.dilate(out_img[z], kernel)
            pass
        for z in range(index, out_img.shape[0] - 1, 1):
            out_img[z + 1] *= cv2.dilate(out_img[z], kernel)
            pass

        out_img = sitk.GetImageFromArray(out_img * 255)
        out_img.CopyInformation(img)
        out_img = sitk.Cast(out_img, sitk.sitkUInt8)
        # out_img = sitk.BinaryDilate(out_img, [2, 2, 0])
        out_img = sitk.Median(out_img, [2, 2, 0])
        # out_img=sitk.BinaryErode(out_img)
        # [229, 125]
        start_index = -100
        end_index = 100
        for x in range(start_index, end_index, 1):
            for y in range(start_index, end_index, 1):
                temp_index = [200 + x, 125 + y]
                if out_img[temp_index[0], temp_index[1], index] == 255:
                    print('找到匹配点')
                    out_img = sitk.ConnectedThreshold(out_img, [temp_index + [index]], 225, 255, 255)
                    sitk.WriteImage(out_img, 'output2.vtk')
                    print('refine-', time.clock() - fineturn_start)
                    print('total-', time.clock() - start)
                    return 0

        sitk.WriteImage(out_img, 'output2.vtk')
        print('refine-', time.clock() - fineturn_start)
    print('total-', time.clock() - start)


if __name__ == '__main__':
    segmentation('/home/bxsh/Liver_data_norm/raw_5.vtk', 4)
