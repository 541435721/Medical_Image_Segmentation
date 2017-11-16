# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  test3.py
# @Date:  2017/8/28 20:01


import scipy.ndimage as sni
import numpy as np
import SimpleITK as sitk
import cv2
from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb
import skimage as ski
import time


def fill_hole(img):
    img_slice = img
    for i in range(img_slice.shape[0]):
        a = 0
        b = img_slice.shape[0] - 1
        for j in range(img_slice.shape[1]):
            if img_slice[i, a] == 0:
                a += 1
            if img_slice[i, b] == 0:
                b -= 1
            if a == b:
                continue
        # print(a, b)
        img_slice[i, a:b] = 1

    for i in range(img_slice.shape[0]):
        a = 0
        b = img_slice.shape[0] - 1
        for j in range(img_slice.shape[1]):
            if img_slice[a, i] == 0:
                a += 1
            if img_slice[b, i] == 0:
                b -= 1
            if a == b:
                continue
        # print(a, b)
        img_slice[a:b, i] = 1
    return img_slice


if __name__ == '__main__':
    img = sitk.ReadImage('out.vtk')

    img_arr = sitk.GetArrayFromImage(img)

    img_arr = np.uint8(img_arr > 0) * 255

    s = time.clock()
    for i in range(img_arr.shape[0]):
        img_arr[i] = fill_hole(img_arr[i])
        print(i)
    print(time.clock() - s)
