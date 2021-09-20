# -*- coding: utf-8 -*-
# 创建日期  : 2021/3/31 15:20 -> ZhangZiyan
# 项目     : 硕士大论文 -> 读excel
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams

FONTSIZE_ZHOU = 14 # 这个控制可能是不起作用的，被上面的覆盖了


config = {
    "font.family":'serif',
    "font.size": FONTSIZE_ZHOU,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)


excel_path = 'E:\数据集\小论文数据_原E盘根目录下\semdata\mos.xlsx'
ex_pd = pd.read_excel(excel_path, header=None)
ex_np = np.array(ex_pd)
ex_np = ex_np.squeeze()


# def trans(num):
#     if 0 < num <= 0.5:
#         num = 0.25
#     elif 0.5 < num <= 1:
#         num = 0.75
#     elif 1 < num <= 1.5:
#         num = 1.25
#     elif 1.5 < num <= 2:
#         num = 1.75
#     elif 2 < num <= 2.5:
#         num = 2.25
#     elif 2.5 < num <= 3:
#         num = 2.75
#     elif 3 < num <= 3.5:
#         num = 3.25
#     elif 3.5 < num <= 4:
#         num = 3.75
#     elif 4 < num <= 4.5:
#         num = 4.25
#     elif 4.5 < num <= 5:
#         num = 4.75
#     return num
a = ex_np
a = a*5
# a = np.sort(a)
#
# for i in range(len(a)):
#     a[i] = trans(a[i])

# 25的时候ok
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# y_axis = range(0, 30, 1)
y_axis = range(0, 25, 1)
# y_axis = range(0, 20, 1)

y_axis_jian = [1,2,3,4,5]

# yy = [6, 12, 18, 24, 30]
yy = [5, 10, 15, 20, 25]
# yy = [4, 8, 12, 16, 20]

# plt.hist(a, histtype='bar')
plt.hist(a, histtype='bar', bins=y_axis)
ax.set_xticks(yy)
ax.set_xticklabels(y_axis_jian)

plt.grid(linestyle='-.')
plt.xlim(3, 26)

ax.set_xlabel('平均主观分值', fontsize=FONTSIZE_ZHOU, fontproperties='SimSun')
ax.set_ylabel('样本个数', fontsize=FONTSIZE_ZHOU, fontproperties='SimSun')
ax.spines['right'].set_color('none')  # 将图像右边的轴设为透明
ax.spines['top'].set_color('none')  # 将图像上面的轴设为透明
# plt.plot(a)

# plt.hist(a)
# plt.plot(a)
plt.savefig('SEM直方图.png', dpi=600, bbox_inches='tight')

plt.show()

