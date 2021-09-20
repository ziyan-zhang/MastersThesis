# -*- coding: utf-8 -*-
# 创建日期  : 2021/3/31 15:33 -> ZhangZiyan
# 项目     : 硕士大论文 -> 读mat
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
# coding:UTF-8
'''
Created on 2015年5月12日
@author: zhaozhiyong
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio


from matplotlib import rcParams
FONTSIZE_ZHOU = 14  # 这个控制可能是不起作用的，被上面的覆盖了

config = {
    "font.family":'serif',
    "font.size": FONTSIZE_ZHOU,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)



dataFile = 'E:\数据集\天舒数据集\\1 live\\dmos_realigned.mat'
data = scio.loadmat(dataFile)['dmos_new']

# path_ = 'E:\数据集\天舒数据集\\3 tid2013\\mos.txt'

# with open(path_, "r") as f:    #打开文件
#     data = f.read()   #读取文件
#
#     data = list(data)
#     plt.plot(data)
#     plt.show()
#
#     print(data)


data = np.sort(data)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

y_axis = range(-3, 117, 6)

zeros = np.all(np.equal(data, 0), axis=0)
data1 = data.squeeze()[~zeros]
data1 = data1.squeeze()

plt.hist(data1, histtype='bar', bins=y_axis)
# plt.hist(data1, histtype='bar')

# plt.hist(data, histtype='bar')
# plt.plot(data)

# plt.xlabel('质量分数')
# plt.ylabel('样本个数')
ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
ax.set_xlabel('差分主观分值', fontsize=FONTSIZE_ZHOU, fontproperties='SimSun')
ax.set_ylabel('样本个数', fontsize=FONTSIZE_ZHOU, fontproperties='SimSun')
plt.grid(linestyle='-.')

plt.savefig('LIVE直方图.png', dpi=600, bbox_inches='tight')

# from ignite
plt.show()