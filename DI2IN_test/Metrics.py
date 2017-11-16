# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  Metrics.py
# @Date:  2017/9/25 13:18

from SimpleITK import ReadImage, GetArrayFromImage
import numpy as np

neighbour = np.zeros([3, 3, 3])
neighbour[1, :, :] = 1
neighbour[:, 1, :] = 1
neighbour[:, :, 1] = 1
neighbour[1, 1, 1] = 0


def read_VTK(file_name: str) -> np.array:
    '''
    读取VTK体数据文件
    :param file_name:VTK文件路径
    :return:nd-array，(z,y,x)
    '''
    img = ReadImage(file_name)
    return GetArrayFromImage(img)


def VOE(pre: np.array, label: np.array):
    '''
    计算重叠率误差
    :param pre:分割图像
    :param label:标签
    :return:误差率，float
    '''
    pre_mask = np.int8(pre > 0)
    label_mask = np.int8(label > 0)
    intersection = pre_mask * label_mask
    union = np.int8(pre_mask + label_mask > 0)
    return 1 - np.sum(intersection) * 1.0 / np.sum(union)


def VD(pre: np.array, label: np.array):
    '''
    相对体积误差
    :param pre:分割图像
    :param label:标签
    :return:误差率，float
    '''
    pre_mask = np.int8(pre > 0)
    label_mask = np.int8(label > 0)
    return (np.sum(pre_mask) - np.sum(label_mask)) * 1.0 / np.sum(label_mask)


def avgD(pre: np.array, label: np.array):
    '''
    计算平均对称面距离
    :param pre:分割图像
    :param label:标签
    :return:误差率，float
    '''
    SR = []
    SG = []
    pre_mask = np.int8(pre > 0)
    label_mask = np.int8(label > 0)
    for z in range(1, pre_mask.shape[0] - 1):
        for y in range(1, pre_mask.shape[1] - 1):
            for x in range(1, pre_mask.shape[2] - 1):
                pre_sum = np.sum(pre_mask[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] * neighbour)
                label_sum = np.sum(label_mask[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] * neighbour)
                if pre_sum < 18 and pre_sum > 0:
                    SR.append([z, y, x])
                if label_sum < 18 and label_sum > 0:
                    SG.append([z, y, x])


    return 0


if __name__ == '__main__':
    a = np.zeros([100, 100, 100])
    a[10:90, 40:80, 30:48] = 1
    b = np.zeros([100, 100, 100])
    b[10:88, 30:79, 29:88] = 1
    print(VOE(a, b))
    print(VD(a, b))
    print(avgD(a, b))

    pass
