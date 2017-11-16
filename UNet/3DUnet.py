# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  3DUnet.py
# @Date:  2017/8/11 14:03


import tensorflow as tf
import numpy as np
import cv2
import SimpleITK as sitk

CLASSES = 2
path1 = '/home/bxsh/Data/PATIENT_DICOM'
path2 = '/home/bxsh/Data/liver'


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
        W = tf.placeholder(dtype=tf.float32, shape=[2])

    root_layers = 64
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

    class_weights = W
    weight_map = tf.multiply(flat_labels, class_weights)
    weight_maps = tf.reduce_sum(weight_map, axis=1)
    weighted_loss = tf.multiply(loss_map + loss1 + loss2, weight_maps)

    loss = tf.reduce_mean(weighted_loss)

    pre_img = tf.argmax(pre, -1)
    ans = tf.equal(pre_img, Y)
    acc = tf.reduce_mean(tf.cast(ans, tf.float32))

    steps = 1000
    g_steps = tf.Variable(0)

    rates = tf.train.exponential_decay(0.2, g_steps, 50, 0.95, staircase=True)
    # train = tf.train.GradientDescentOptimizer(rates).minimize(loss, global_step=g_steps)
    train = tf.train.MomentumOptimizer(
        learning_rate=rates, momentum=0.2).minimize(loss=loss, global_step=g_steps)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged = tf.summary.merge_all()


def read_dcm(names):
    names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(names)
    img = sitk.ReadImage(names)
    return sitk.GetArrayFromImage(img)


if __name__ == '__main__':
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        Train_Data = read_dcm(path1)
        Train_Data = Train_Data.reshape([1] + list(Train_Data.shape) + [1])
        Label_Data = read_dcm(path2)
        Label_Data = Label_Data.reshape([1] + list(Label_Data.shape)) / 255.0
        tf.global_variables_initializer().run()
        batch_size = 4
        key = 0.0045  # 0.005
        summary_writer = tf.summary.FileWriter('./summary', graph=sess.graph)

        x = Train_Data[:, 100:104, ...]
        y = Label_Data[:, 100:104, ...]
        w = [0.1, 0.2]
        print(x.shape)
        print(y.shape)
        ans3 = 1000
        pic = sess.run(pre, feed_dict={X: x, Y: y, W: w})
        # cv2.imwrite('./prediction/test_.jpg', np.uint8(
        # (pic[0, :, :, 0] < pic[0, :, :, 1])) * 255)
        ans1, ans2, ans3, ans4 = sess.run(
            [loss, acc, rates, merged], feed_dict={X: x, Y: y, W: w})
        print('acc:' + str(ans2) + ' loss:' + str(ans1))
        for iters in range(10000):
            batch_size += iters/5000
            for i in range(0, Train_Data.shape[1]-batch_size, 1):
                x = Train_Data[:,i:i + batch_size,...]
                y = Label_Data[:,i:i + batch_size,...]
                flat = y.flatten().tolist()
                ans1, ans2, ans3, ans4 = sess.run(
                    [loss, acc, rates, merged], feed_dict={X: x, Y: y, W: w})
                sess.run(train, feed_dict={X: x, Y: y, W: w})
                summary_writer.add_summary(ans4, iters + i)
                print("Iteration:%s,loss:%f,acc:%f,rates:%f"%(str(iters) + "-" + str(i), ans1, ans2, ans3))
                pic = sess.run(pre_img, feed_dict={X: x, Y: y, W: w})
                print(pic.shape)
                cv2.imwrite('./prediction/pre_' + str(iters) + '_' + str(i) + '.jpg',
                           np.uint8(pic[0,0,...]) * 255)
            if ans3-0<0.00001:
                break
                # cv2.imwrite('./prediction/gt_' + str(iters) + '_' + str(i) + '.jpg', y[0,0] * 255)

        saver.save(sess, './test_model_save/test.ckpt')
