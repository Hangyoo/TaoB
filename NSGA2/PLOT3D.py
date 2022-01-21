import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot(front4):
    #定义坐标轴
    front4 = np.array(front4)
    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')

    xd = [item[1] for item in front4]
    yd = [item[0] for item in front4]
    zd = [item[2] for item in front4]
    ax1.scatter3D(xd,yd,zd, c='b', marker='o', s=15)  #绘制散点图

    ax1.set_xlabel("TN",fontsize=13, labelpad=8)
    ax1.set_ylabel("TP",fontsize=13, labelpad=8)
    ax1.set_zlabel("COST",fontsize=13, labelpad=10)
    # 调整坐标轴刻度大小
    plt.tick_params(labelsize=12,pad=5)
    ax1.view_init(azim=-107, elev=23)
    plt.show()

