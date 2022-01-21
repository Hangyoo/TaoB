import numpy as np
import geatpy as ea


# 定义常量
k1 = 0.21
k2 = 1.518
alpha = 0.81
delta = 0.3

# 2010 - 2019年平均数据
GDP = 30851.096
FA = 155961.1921
EFA = 7361.89476
UEC = 0.56746
C1 = 0.7759
C2 = 0.5857
C3 = 0.4483
E1 = 7666.985684
E2 = 3442.045447
E3 = 2532.619504

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 4  # 初始化Dim（决策变量维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0,0,0.016,5.5]  # 决策变量下界
        ub = [0.85,0.2,0.1,6]  # 决策变量上界
        lbin = [0,0,0,0]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [0,0,0,0]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        x3 = Vars[:, [2]]
        x4 = Vars[:, [3]]

        # 不等式约束条件：
        CV1 = x2 - x1
        CV2 = x1 + alpha * x3 - 1
        CV3 = (C1*E1+C2*E2+C3*E3 - 390000) * x1  # C1*E1+C2*E2+C3*E3 <= 390000  这里乘的x1是为了程序运行，不影响结果

        pop.CV = np.hstack([CV1, CV2,CV3])

        f1 = - k1 * (GDP * x1 + FA)

        f2 = np.abs(k2 * (GDP*x2+EFA) - GDP*UEC)

        f3 = C1*E1+C2*E2+C3*E3 - (alpha*x3*GDP)*x4

        f = 0.3 * f1 + 0.2 *f2 + 0.5 *f3

        pop.ObjV = np.hstack([f])  # 把求得的目标函数值赋值给种群pop的ObjV

    def calObj(self, pop):  # 目标函数

        x1 = pop[0]
        x2 = pop[1]
        x3 = pop[2]
        x4 = pop[3]

        f1 = - k1 * (GDP * x1 + FA)

        f2 = abs(k2 * (GDP*x2+EFA) - GDP*UEC)

        f3 = C1*E1+C2*E2+C3*E3 - (alpha*x3*GDP)*x4

        f = 0.1 * f1 + 0.4 *f2 + 0.5 *f3

        print('目标函数f1的值为:',-f1)
        print('目标函数|f2|的值为:',f2)
        print('目标函数f3的值为:',f3)
        # print('目标函数加权和为:',f)