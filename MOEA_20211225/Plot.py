import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
    功能：汇总3中算法结果，绘制pareto最优解
'''

#定义坐标轴
fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

front = pd.read_csv("./Result/Objective_NSGA3.CSV")
xd = np.array(front.iloc[:,0]).tolist()
yd = np.array(front.iloc[:,1]).tolist()
zd = np.array(front.iloc[:,2]).tolist()
ax1.scatter3D(xd,yd,zd, c='green', marker='x', s=15)  #绘制散点图

front = pd.read_csv("./Result/Objective_MOEAD.CSV")
xd = np.array(front.iloc[:,0]).tolist()
yd = np.array(front.iloc[:,1]).tolist()
zd = np.array(front.iloc[:,2]).tolist()
ax1.scatter3D(xd,yd,zd, c='red', marker='o', s=15)  #绘制散点图

front = pd.read_csv("./Result/Objective_NSGA2.CSV")
xd = np.array(front.iloc[:,0]).tolist()
yd = np.array(front.iloc[:,1]).tolist()
zd = np.array(front.iloc[:,2]).tolist()
ax1.scatter3D(xd,yd,zd, c='blue', marker='+', s=15)  #绘制散点图

ax1.legend(["NSGA3","MOEA/D",'NSGA2'])

ax1.set_xlabel("F1")
ax1.set_ylabel("F2")
ax1.set_zlabel("F3")

plt.show()
