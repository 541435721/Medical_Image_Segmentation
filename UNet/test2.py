# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  test2.py
# @Date:  2017/8/28 20:42


import SimpleITK as sitk
import numpy as np

path1 = 'C:/Users/wow00/Desktop/Data/PATIENT_DICOM'
path2 = 'C:/Users/wow00/Desktop/Data/liver'

raw_path_list = [path1]
label_path_list = [path2]


def read_dicom(path):
    names = sitk.ImageSeriesReader_GetGDCMSeriesFileNames(path)
    img = sitk.ReadImage(names)
    return img


if __name__ == '__main__':
    img = read_dicom(path1)
    label = read_dicom(path2)
    mask = sitk.Threshold(img, 102 - 3 * 43, 102 + 3 * 43, -1024)

    img = sitk.IntensityWindowing(mask, 102 - 3 * 43, 102 + 3 * 43, 0, 255)
    img = sitk.Median(img)
    temp_arr = sitk.GetArrayFromImage(img)
    temp_mask = np.zeros_like(temp_arr)
    for i in range(1, temp_arr.shape[0]):
        temp_mask[i] = abs(np.int16(temp_arr[i - 1]) - np.int16(temp_arr[i]))
        # print(np.sum(temp_mask[i]))

    label_mask = sitk.GetImageFromArray(temp_mask)
    label_mask.CopyInformation(label)

    sitk.WriteImage(label_mask, 'label_mask.vtk')
