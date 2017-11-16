# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  Data_Generator.py
# @Date:  2017/9/26 13:32

from SimpleITK import ReadImage, GetArrayFromImage
import numpy as np
import os
import time

X_path = []
Y_path = []

path = '/home/bxsh/Liver_data'


def read_VTK(file_name):
    '''
    读取VTK体数据文件
    :param file_name:VTK文件路径
    :return:nd-array，(z,y,x)
    '''
    img = ReadImage(file_name)
    return GetArrayFromImage(img)


class Data:
    def __init__(self, path, block_size):
        self.root_path = path
        self.BLOCK_SIZE = block_size
        files = os.listdir(self.root_path)
        raw_data_filename = list(filter(lambda x: x.startswith('raw'), files))
        label_data_filename = list(filter(lambda x: x.startswith('label'), files))
        self.data_filename = list(zip(raw_data_filename, label_data_filename))
        self.gen = self.generator()

    def generator(self):
        for raw, label in self.data_filename:
            label_ = raw.replace('raw','label')
            raw_data = read_VTK(self.root_path + '/' + raw)
            label_data = read_VTK(self.root_path + '/' + label_)
            raw_data = raw_data.reshape([1] + list(raw_data.shape) + [1])
            label_data = 1.0 * (label_data.reshape([1] + list(label_data.shape)) > 0)

            for z in range(0, raw_data.shape[1] - self.BLOCK_SIZE[0]-1, 20):
                for y in range(0, raw_data.shape[2] - self.BLOCK_SIZE[1]-1, 50):
                    for x in range(0, raw_data.shape[3] - self.BLOCK_SIZE[2]-1, 50):
                        R = raw_data[:, z:z + self.BLOCK_SIZE[0], y:y + self.BLOCK_SIZE[1], x:x + self.BLOCK_SIZE[2], :]
                        L = label_data[:, z:z + self.BLOCK_SIZE[0], y:y + self.BLOCK_SIZE[1], x:x + self.BLOCK_SIZE[2]]
                        yield R, L

    def next(self):
        return self.gen.next()


if __name__ == '__main__':
    data = Data(path, BLOCK_SIZE)
    count = 0
    while True:
        count += 1
        try:
            x, y = data.next()
            print x, y
        except Exception as e:
            data = Data(path, BLOCK_SIZE)
