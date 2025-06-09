import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd

#看数据集值的分布
X= pd.read_csv('mat.csv',index_col=0)
data = X.to_numpy()

# 将矩阵展平为一维数组，方便绘制直方图
flattened_speed_data = data.flatten()

# 绘制直方图
plt.figure(figsize=(8, 6))  # 设置画布大小
plt.hist(flattened_speed_data, bins=50, color='blue', alpha=0.7,density=True) # 设置直方图的 bins 数量和颜色
plt.title('SEDATA',fontsize=16) # 设置标题
plt.xlabel('Speed (km/h)',fontsize=16) # 设置 x 轴标签
plt.ylabel('Proportion',fontsize=16) # 设置 y 轴标签
# plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.show() # 显示图像
