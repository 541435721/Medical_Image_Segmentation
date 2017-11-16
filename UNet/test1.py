# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  test1.py
# @Date:  2017/8/24 8:44


from numpy import *
import cv2
from skimage.segmentation import slic, mark_boundaries, active_contour
import numpy as np
import SimpleITK as sitk
import time

path1 = 'C:/Users/wow00/Desktop/Data/PATIENT_DICOM'
path2 = 'C:/Users/wow00/Desktop/Data/liver'

raw_path_list = [path1]
label_path_list = [path2]


def read_dicom(path):
    names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path)
    img = sitk.ReadImage(names)
    return img


def get_prob(raw_path_list, label_path_list):
    layer = []
    map_x_y = np.zeros([512, 512])
    map_itensity = np.zeros([512])
    for raw_path, label_path in zip(raw_path_list, label_path_list):
        img = read_dicom(raw_path)
        label = read_dicom(label_path)

        img = sitk.IntensityWindowing(img, 102 - 3 * 43, 102 + 3 * 43, 0, 255)

        img = np.uint8(sitk.GetArrayFromImage(img))
        label = np.uint8(sitk.GetArrayFromImage(label) > 0)
        layer.append(label.shape[0])

        map_x_y += np.sum(label, axis=0) * 1.0

        img = img * label

        img = img[img > 0]
        for i in img:
            map_itensity[i] += 1
    map_x_y = map_x_y / 512.0 / 512 / sum(layer)
    map_itensity = map_itensity / 512.0 / 512 / sum(layer)
    np.save('map_x_y', map_x_y)
    np.save('map_itensity', map_itensity)


def calc_new_map(img, map_x_y, map_itensity):
    normal_map_x_y = (map_x_y - map_x_y.min()) / (map_x_y.max() - map_x_y.min())
    normal_map_itensity = (map_itensity - map_itensity.min()) / (map_itensity.max() - map_itensity.min())
    normal_map_x_y = [normal_map_x_y for i in range(img.shape[0])]
    normal_map_x_y = np.stack(normal_map_x_y, axis=0)
    temp = np.array([normal_map_itensity[i] for i in img.flat]).reshape(img.shape)
    map = normal_map_x_y * temp
    return map


if __name__ == '__main__':
    start = time.clock()
    map_x_y = np.load('./median/map_x_y.npy')
    map_itensity = np.load('./median/map_itensity.npy')[:256]

    cv2.imshow('1', map_x_y * 100000)
    cv2.waitKey()
    s = time.clock()
    raw = read_dicom(path1)
    print(time.clock() - s)
    img = sitk.IntensityWindowing(raw, 102 - 3 * 43, 102 + 3 * 43, 0, 255)
    img = sitk.GetArrayFromImage(img)
    # img = sitk.Median(img)

    t = np.linspace(0.1, 0.9, 100)  # 21

    map = calc_new_map(img, map_x_y, map_itensity)

    map = np.uint8(map > t[21])

    layer_sum = [np.sum(map[i, :, :]) for i in range(map.shape[0])]
    index = layer_sum.index(max(layer_sum))
    print(index)  # 101
    # cv2.imshow('test1', map[index] * 255)
    # map[index] = cv2.dilate(map[index], (3, 3), 1)
    # map[index] = cv2.erode(map[index], (3, 3), 1)
    # init = [[i, j] for i in range(0, 512, 511) for j in range(512)]
    # init.extend([i, j] for i in range(512) for j in range(0, 512, 511))
    # init = np.array(init)
    # segments = np.int32(active_contour(map[index], init, w_edge=0, w_line=1))
    # for i in range(segments.shape[0]):
    #     print(index, segments[i, 0], segments[i, 1])
    #     map[index, segments[i, 0], segments[i, 1]] = 255
    # print(segments.shape)
    print(time.clock() - start)
    cv2.imshow('test2', map[index] * 255)
    cv2.waitKey()
    '''
    for i in range(index + 1, map.shape[0]):
        new_mask = cv2.dilate(map[i - 1], (3, 3))
        new_mask = cv2.erode(new_mask, (3, 3), 2)
        new_mask *= map[i]
        map[i] = new_mask   
    '''
    out = sitk.GetImageFromArray(map)
    out.CopyInformation(raw)
    sitk.WriteImage(sitk.Cast(out, raw.GetPixelID()) * raw, 'out.vtk')

    # img_slice = img[100]
    # segments = slic(img_slice, n_segments=50, compactness=0.5)
    # cv2.imshow('test1', img_slice)
    # cv2.imshow('test2', mark_boundaries(img_slice, segments))
    # print(segments.max(), segments.min())
    # cv2.waitKey()
