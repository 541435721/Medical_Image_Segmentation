# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  predict_all.py
# @Date:  2017/9/29 9:31


from D3Unet_modify3_pre import predict, read_dcm
import os

root_path = '/home/bxsh/Liver_data'

if __name__ == '__main__':
    filenames = os.listdir(root_path)
    for file in filenames:
        if file.startswith('raw'):
            print(file)
            vtk_name = root_path + '/' + file
            predict(vtk_name, out_file=str(file) + '.vtk')
    pass
