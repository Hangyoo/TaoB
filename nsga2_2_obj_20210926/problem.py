import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 = C0
min f2 = C

s.t.
40 <= d <= 60
40 <= L <= 100
30 <= Z <= 70
893 <= D <= 1007
20 <= alpha <= 25
1 <= fi <= 2
"""

# 定义常量
i = 1
Lambda = 0.4
K_Dmin = 0.3
K_Dmax = 0.5
e = 0.03
beta = 0.8
epsilon = 0.4
d0 = 1090
di = 810
C = 126


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 6  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [40,40,30,893,20,1]  # 决策变量下界
        ub = [60,100,70,1007,25,2]  # 决策变量上界
        lbin = [1,1,1,1,1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        d = Vars[:, [0]]
        L = Vars[:, [1]]
        Z = Vars[:, [2]]
        D = Vars[:, [3]]
        alpha = Vars[:, [4]]
        fi = Vars[:, [5]]

        # 不等式约束条件：
        CV1 = K_Dmin*(d0-di)/np.cos(alpha* np.pi/180) - 2*d
        CV2 = 2*d - K_Dmax*(d0-di)/np.cos(alpha* np.pi/180)
        CV3 = (0.5-e)*(d0+di) - D
        CV4 = D - (0.5+e)*(d0+di)
        CV5 = L - beta*C/np.cos(alpha* np.pi/180)
        CV6 = d - L
        CV7 = epsilon*d - 0.5*(d0-d-D)/np.cos(alpha* np.pi/180)
        CV8 = Z*np.pi/180 + 2*Z*(1/np.sin(d/D*np.cos(alpha* np.pi/180)* np.pi/180)) - 2*np.pi
        CV9 = Z - np.pi*D / (d+L*np.sin(fi)* np.pi/180)

        pop.CV = np.hstack([CV1, CV2, CV3, CV4, CV5, CV6, CV7,  CV9])


        f1 = 44*(1-d/D*np.cos(alpha* np.pi/180))*Z*L*d*np.cos(alpha* np.pi/180)

        f2 = 207*Lambda*(1+(1.04*((1-d/D*np.cos(alpha* np.pi/180))/(1+d/D*np.cos(alpha* np.pi/180)))**(143/108))**(9/2))**(-2/9)*\
             ((d/D*np.cos(alpha* np.pi/180))**(2/9)*(1-d/D*np.cos(alpha* np.pi/180))**(29/27))/((1+d/D*np.cos(alpha* np.pi/180))**(1/4))*\
             (i*L*np.cos(alpha* np.pi/180))**(7/9)*Z**(3/4)*d**(29/27)

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV