# -*- coding: utf-8 -*-
# 创建日期  : 2021/3/15 21:22 -> ZhangZiyan
# 项目     : 硕士大论文 -> 可视化不同分类的rank
# 描述     :  
# 待办     :  
__author__ = 'ZhangZiyan'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
FONTSIZE_ZHOU = 12  # 这个控制可能是不起作用的，被上面的覆盖了

LabelPath = 'mos.xlsx'

labelFpath = LabelPath
label_pd = pd.read_excel(labelFpath, header=None)
label_array = np.array(label_pd)
label_array = label_array.squeeze()

# for i in range(50):
#     for j in range(13):
#         plt.scatter(j, label_array[13*i + j])
# plt.savefig('示意图.png', dpi=600, bbox_inches='tight')
# plt.show()

label_array_ori = label_array

label_array = label_array.reshape(-1, 13)
label_df = pd.DataFrame(label_array)

fig, ax = plt.subplots(figsize=[10, 6])
# fig, ax = plt.subplots()

sns.boxplot(data=label_df)

ax.set_xlabel('图像失真类型', fontsize=FONTSIZE_ZHOU+2, fontproperties='SimSun')
ax.set_ylabel('主观意见分数', fontsize=FONTSIZE_ZHOU+2, fontproperties='SimSun')

# x_ticks = ax.set_xticks(x_ticks)
# xlabels = ax.set_xticklabels(rotation=0, fontsize=FONTSIZE_ZHOU,
#                              fontproperties='Times New Roman')
x_ticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
x_ticks = ax.set_xticks(x_ticks)
xlabels = ax.set_xticklabels(['较高质', '噪声轻', '噪声中', '噪声重', '模糊轻', '模糊中', '模糊重', '过亮', '过暗', '对比1', '对比2',
                              '像散1', '像散2'], rotation=0, fontsize=FONTSIZE_ZHOU, fontproperties='SimSun')

plt.savefig('示意图.png', dpi=600, bbox_inches='tight')

# for i in range(50):
#     for j in range(13):
#         plt.scatter(j, label_array[13*i + j])
# plt.savefig('示意图.png', dpi=600, bbox_inches='tight')
# plt.show()


for i in range(5, 10):
    plt.plot(label_array_ori[13*i: 13*(i+1)], label=str(i))

# i = 13
# plt.plot(label_array_ori[13*i: 13*(i+1)], label=str(i))


plt.legend()
plt.show()