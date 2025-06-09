import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd

#看数据集值的分布

stations_data = pd.read_excel('stations.xlsx').to_numpy()
stations_data = stations_data[np.lexsort(stations_data[:, ::-1].T)]
stations_list = stations_data[:, 0]
# G = nx.Graph()
# G.add_nodes_from(stations_list)
# for i in range(0, len(stations_list)):
#     for j in range(i + 1, len(stations_list)):
#         G.add_edge(stations_list[i], stations_list[j],
#                    weight=haversine(stations_data[i][1], stations_data[i][2], stations_data[j][1],
#                                          stations_data[j][2]))
# A = nx.adjacency_matrix(G).todense()

# Gaussian kernel
# original_A = A
# A = np.exp(- 0.5 * (original_A / np.std(original_A, axis=1, keepdims=True)) ** 2)

# here we use interpolate
aq_data = []
for path in  ['beijing_201802_201803_aq.csv', 'beijing_17_18_aq.csv']:
    aq_data.append(pd.read_csv(path))
# interpolate missing values
beijing_aq = pd.concat(aq_data).sort_values(
    by=['stationId', 'utc_time']).drop(columns=['stationId', 'utc_time'])
missing_index = beijing_aq.isna().to_numpy().reshape((len(stations_list), -1, 6))
# beijing_aq = beijing_aq.interpolate('linear', axis=0).ffill().bfill().fillna(0)
beijing_aq = beijing_aq.fillna(0)

beijing_aq = beijing_aq.to_numpy().reshape((len(stations_list), -1, 6))
beijing_time = pd.concat(aq_data).sort_values(by=['stationId', 'utc_time'])['utc_time'].unique().tolist()

# PM 2.5
data = beijing_aq[..., 0]
missing_index = missing_index[..., 0]


# 将矩阵展平为一维数组，方便绘制直方图
flattened_speed_data = data.flatten()

# 绘制直方图
plt.figure(figsize=(8, 6))  # 设置画布大小
plt.hist(flattened_speed_data, bins=50,range=(0, 500), color='blue', alpha=0.7,density=True) # 设置直方图的 bins 数量和颜色
plt.title('BJAIR',fontsize=16) # 设置标题
plt.xlabel('Speed (km/h)',fontsize=16) # 设置 x 轴标签
plt.ylabel('Proportion',fontsize=16) # 设置 y 轴标签
# plt.grid(True, linestyle='--', alpha=0.5)  # 添加网格线
plt.show() # 显示图像
