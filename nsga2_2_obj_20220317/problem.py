import numpy as np
import geatpy as ea
import math

"""
最小化目标3目标优化问题
min f1 
min f2 
min f3 
"""

# 定义常量
alpha = 42
ro = 840
P = 10
R = 0.3
beta = 17 * (math.pi / 180) # 角度弧度制转换
A = 17*17*math.pi/4


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 12  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0,0,0,0,351,4,180,171,173,184,0,355]  # 决策变量下界
        ub = [0.3,0.3,0.3,0.3,360,8,188,177,182,188,8,360]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        m1 = Vars[:, [0]]
        m2 = Vars[:, [1]]
        m3 = Vars[:, [2]]
        m4 = Vars[:, [3]]
        # 角度制都需要转为弧度制
        s1 = Vars[:, [4]] * np.pi / 180
        e1 = Vars[:, [5]] * np.pi / 180
        s2 = Vars[:, [6]] * np.pi / 180
        e2 = Vars[:, [7]] * np.pi / 180
        s3 = Vars[:, [8]] * np.pi / 180
        e3 = Vars[:, [9]] * np.pi / 180
        s4 = Vars[:, [10]] * np.pi / 180
        e4 = Vars[:, [11]] * np.pi / 180


        f1 = alpha * np.sqrt(2*P/ro) * (m1*(e1-s1+math.pi) - m4*(s4-2*e4+math.pi+s1))

        f2 = alpha * np.sqrt(2*P/ro) * (m3*(e3-s3) - m2*(s2-2*e2+s3))

        f3 = R/(np.cos(beta)**2) * A * P * (np.cos(s1)+np.cos(s1+40*np.pi/180)+np.cos(s1+2*40*np.pi/180)+np.cos(s1+3*40*np.pi/180)+np.cos(s1+4*40*np.pi/180)
            - (np.cos(s2)+np.cos(s2-40*np.pi/180)+np.cos(s2-2*40*np.pi/180)+np.cos(s2-3*40*np.pi/180)+np.cos(s2-4*40*np.pi/180)))

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV