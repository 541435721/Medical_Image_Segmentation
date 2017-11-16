# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  data_preprocess.py
# @Date:  2017/11/15 9:17


import os
import SimpleITK as sitk
import numpy as np

path = 'C:\\Users\\wow00\\Desktop\\Training\\'.replace('\\', '/')


def read_dicom(path):
    # print(path)
    names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path)
    img = sitk.ReadImage(names)
    return img


def pre_process(img, out='test.vtk'):
    # -967.760272674 177.979353105
    mean = -967.760272674
    std = 177.979353105
    image = sitk.Cast(img, sitk.sitkFloat32)
    image = sitk.IntensityWindowing(image, mean - 3 * std, mean + 3 * std, 0.0, 255.0)
    sitk.WriteImage(image, out)


def calc_mean_std(raw, label):
    raw_arr = sitk.GetArrayFromImage(raw)
    label_arr = sitk.GetArrayFromImage(label) > 0
    selected_arr = raw_arr[label_arr]
    airway_mean, airway_std = np.mean(selected_arr), np.std(selected_arr)
    return airway_mean, airway_std


def statistic(path):
    paths = os.listdir(path)
    means = []
    stds = []
    for dir_name in paths:
        print('正在处理：', dir_name)
        raw_path = path + dir_name + '/raw'
        label_path = path + dir_name + '/label'
        raw = read_dicom(raw_path)
        label = read_dicom(label_path)
        a, b = calc_mean_std(raw, label)
        means.append(a)
        stds.append(b)
    return np.mean(means), max(stds)


def pre_process_all(path):
    paths = os.listdir(path)
    count = 1
    for dir_name in paths:
        print('正在处理：', dir_name)
        raw_path = path + dir_name + '/raw'
        label_path = path + dir_name + '/label'
        raw = read_dicom(raw_path)
        label = read_dicom(label_path)
        pre_process(raw, 'raw_' + str(count) + '.vtk')
        sitk.WriteImage(label, 'label_' + str(count) + '.vtk')
        count += 1


if __name__ == '__main__':
    # a, b = statistic(path)  # -967.760272674 177.979353105
    pre_process_all(path)
    pass
