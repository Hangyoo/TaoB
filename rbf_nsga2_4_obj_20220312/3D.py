import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义坐标轴
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

reference_front_path = r"C:\Users\Hangyu\PycharmProjects\TaoB\rbf_nsga2_4_obj_20220312\Population Info\ObjV.csv"
front1 = pd.read_csv(reference_front_path,header=None)



xd = list(front1.iloc[:,1])
yd = list(front1.iloc[:,2])
zd = list(front1.iloc[:,3])
ax1.scatter3D(xd,yd,zd, color='b', marker='o',s=15)  #绘制散点图

ax1.set_xlabel("ha",fontsize=12)
ax1.set_ylabel("Haz D",fontsize=12)
ax1.set_zlabel("De",fontsize=12)

# 调整坐标轴刻度大小
plt.tick_params(labelsize=12,pad=0)

plt.show()