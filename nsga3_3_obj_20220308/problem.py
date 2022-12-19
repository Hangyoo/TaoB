import pandas as pd
import warnings
import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 
min f2 
min f3 
"""
Fvi = [17.21 for _ in range(37)]
Fsi = [18.8 for _ in range(37)]
Fri = [1.76 for _ in range(37)]
alpha = [0 for _ in range(27)] + [6.5,13,19.5,26,32.5,39.5,46.5,54,31.5,68.5]
mu = [0.25 for _ in range(37)]
Ri = [0.216,0.216,0.216,0.216,0.216] + [0.2415 for i in range(31)]
h = [0.005 for _ in range(37)]


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=3):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 74  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.35 for i in range(37)] + [0 for i in range(37)]  # 决策变量下界
        ub = [2.74 for i in range(37)] + [360 for i in range(37)]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=float)

        ro = Vars[:, 0:37]
        theta = Vars[:, 37:]


        Mx = [Fvi[i]*ro[:,[i]]*np.cos(alpha[i])*np.sin(theta[i])+Fsi[i]*ro[:,[i]]*np.sin(alpha[i])*np.sin(theta[i]) for i in range(37)]

        My =0
        f1 = np.sqrt(Mx**2 + My**2)

        # Fx =
        # Fy =
        # f2 = np.sqrt(Fx**2 + Fy**2)
        f2 = f1
        #
        # w_bar = np.sum([(-mu[i]**2*np.pi* (np.cos(Ri[i]-h[i])**-1)*Fvi[i]*ro[:,[i]]*(1-0.5*(1-Fri[i]/mu[i]*Fri[i]))) for i in range(37)])/37
        f3 = f1

        pop.ObjV = np.hstack([f1, f2, f3])  # 把求得的目标函数值赋值给种群pop的ObjV

        print(Mx)

        Constr = []
        for i in range(36):
            Cv1 = ro[:,[i+1]] - ro[:,[i]] - 0.08
            Cv2 = 0.08 - ro[:,[i+1]] + ro[:,[i]]
            Constr.extend([Cv1,Cv2])

        constr1 = sum(ro[:,[i]] * np.cos(theta[:,[i]]) for i in range(37))/37 - 20
        constr1 = sum(ro[:, [i]] * np.sin(theta[:, [i]]) for i in range(37)) / 37 - 20
        pop.CV = np.hstack(Constr)
