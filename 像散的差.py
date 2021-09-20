# -*- coding: utf-8 -*-
# 创建日期  : 2021/3/12 11:05 -> ZhangZiyan
# 项目     : 所有绘图 -> difference
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

img1_path = 'difference/02_001.tif'
img2_path = 'difference/02_012.tif'

img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), 0)
img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), 0)

img3 = img1 - img2
figName = 'difference/astigmatism_diff.png'

cv2.imwrite(figName, img3)
