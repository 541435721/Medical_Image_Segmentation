# -*- coding: utf-8 -*-
# @Author: bxsh
# @Email:  xbc0809@gmail.com
# @File:  test.py
# @Date:  2017/8/24 13:53

import cv2
import numpy as np
from skimage.draw import circle_perimeter
from skimage.segmentation import active_contour
from skimage.filters import gaussian

if __name__ == '__main__':
    # a = np.uint8(np.zeros([300, 300]))
    # a[50:-50, 50:-50] = 1
    # a[149:-149, 149:-149] = 0
    # b = cv2.erode(a, kernel=(3, 3), iterations=5)
    # cv2.imshow('a', a * 255)
    # cv2.imshow('b', b * 255)
    # cv2.waitKey()
    img = np.zeros((100, 100))
    rr, cc = circle_perimeter(35, 45, 25)
    img[rr, cc] = 1
    img = gaussian(img, 2)
    s = np.linspace(0, 2 * np.pi, 100)
    init = 50 * np.array([np.cos(s), np.sin(s)]).T + 50
    snake = active_contour(img, init, w_edge=0, w_line=1)
    print(snake.shape)
pass
