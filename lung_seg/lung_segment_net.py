# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  lung_segment_net.py
# @Date:  2017/11/9 14:17




from tf_unet import unet, util, image_util
import numpy as np
from tf_unet.image_util import SimpleDataProvider
import SimpleITK as sitk
from SimpleITK import ImageSeriesReader_GetGDCMSeriesFileNames, ReadImage, GetArrayFromImage, WriteImage
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_dicom(path, label=False):
    name = ImageSeriesReader_GetGDCMSeriesFileNames(path)
    img = ReadImage(name)
    if label:
        arr = GetArrayFromImage(img)
        arr = arr > 2000
    else:
        img = sitk.Cast(img, sitk.sitkFloat32)
        img = sitk.IntensityWindowing(img, -1024, 1024, 0.0, 1.0)
        # WriteImage(img, 'test.vtk')
        arr = GetArrayFromImage(img)
        arr = arr[..., np.newaxis]
    print(arr.shape)
    return arr


model_path = './lung_segment_model'
trained_model = './lung_segment_model_1/model.cpkt'


def lung_segment(path, model_path):
    name = ImageSeriesReader_GetGDCMSeriesFileNames(path)
    raw = ReadImage(name)
    raw = sitk.Cast(raw, sitk.sitkFloat32)
    raw = sitk.IntensityWindowing(raw, -1024, 1024, 0, 1.0)
    arr = GetArrayFromImage(raw)
    prediction = np.zeros_like(arr)
    arr = arr[..., np.newaxis]
    net = unet.Unet(layers=3, features_root=32, channels=1, n_class=2, summaries=False)
    pre = net.predict(model_path, arr, 4)
    pre = np.argmax(pre, -1)
    prediction[:, 20:492, 20:492] = pre

    stride = 50
    index = None
    for z in range(0, prediction.shape[0], stride):
        for y in range(154, prediction.shape[1], stride):
            for x in range(105, prediction.shape[2], stride):
                patch = prediction[z:z + stride, y:y + stride, x:x + stride]
                ratio = patch.mean()
                if ratio > 0.95:
                    index = [z + stride // 2, y + stride // 2, x + stride // 2]
                    break
            if index:
                break
        if index:
            break
    index.reverse()
    # print(index)
    prediction = sitk.GetImageFromArray(prediction)
    prediction.CopyInformation(raw)
    prediction = sitk.Cast(prediction, sitk.sitkUInt8)
    prediction = sitk.ConnectedThreshold(prediction, [index], 1, 1, 1)
    return prediction


def train(raw_path, label_path, model_path):
    data = read_dicom(raw_path)
    label = read_dicom(label_path, True)
    # 创建训练集
    data_provider = SimpleDataProvider(data, label, n_class=2, channels=1)

    # 构建网络
    net = unet.Unet(layers=3, features_root=32, channels=1, n_class=2, summaries=False)
    trainer = unet.Trainer(net, batch_size=2, opt_kwargs={'learning_rate': 0.02})
    path = trainer.train(data_provider, model_path, training_iters=64, epochs=100)
    print(path)


if __name__ == '__main__':
    # train('./Lung/lung_raw', './Lung/lung_label/all',model_path)
    pre = lung_segment('./Lung/lung_raw', trained_model)
    sitk.WriteImage(pre, 'test.vtk')
