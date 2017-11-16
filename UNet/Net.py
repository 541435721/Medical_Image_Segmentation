# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  Net.py
# @Date:  2017/8/24 20:04

import cv2
import numpy as np
import SimpleITK as sitk
from tf_unet import unet, util, image_util

Model_PATH = './Models'
path1 = '/home/bxsh/Data/PATIENT_DICOM'
path2 = '/home/bxsh/Data/liver'


def read_dcm(path, label=False):
    if label:
        names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path2)
        image = sitk.ReadImage(names)
    else:
        image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)
    # image = image.transpose([1, 0, 2])
    if label:
        image = image > 0
    else:
        image = image.reshape(list(image.shape) + [1]) / 255.0
    return image


if __name__ == '__main__':
    Train_Data = read_dcm('out.vtk')
    Label_Data = read_dcm(path2, True)
    print(Train_Data.shape, Label_Data.shape)
    # 加载数据
    data_provider = image_util.SimpleDataProvider(
        Train_Data, Label_Data, n_class=2, channels=1)
    # 创建并训练网络
    net = unet.Unet(layers=3, features_root=64, channels=1, n_class=2)
    # trainer = unet.Trainer(net, batch_size=2)
    # path = trainer.train(data_provider, Model_PATH, training_iters=32, epochs=50, write_graph=True)
    
    for i in range(3,5):
        trainer = unet.Trainer(net, batch_size=i)
        path = trainer.train(data_provider, Model_PATH, training_iters=32, epochs=100, write_graph=True,restore=True)
        print(path)
    # print(data_provider.n_class)
    pass
