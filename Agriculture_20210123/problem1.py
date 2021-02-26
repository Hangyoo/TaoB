# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:03:57 2021

@author: 樊煜
"""

import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f
min S

s.t.
0<= x1-x11 <= 2.5  流量
0<= x12-x22 <= 25  开始时间
0<= x23-x33 <= 25    结束时间
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 33  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * 11+ [1] * 22  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        #lb = [0] * Dim  # 决策变量下界
        lb = [0.36, 0.6, 0.6, 0.36, 0.36, 0.3, 0.3, 0.36, 0.48, 0.9, 0.48]+[0]*22
        ub = [0.72, 1.2, 1.2, 0.72, 0.72, 0.6, 0.6, 0.72, 0.96, 1.8, 0.96]+[25]*22
        #lb = [36, 60, 60, 36, 36, 30, 30, 36, 48, 90, 48]+[0]*22
        #ub = [72, 120, 120, 72, 72, 60, 60, 72, 96, 180, 96]+[25]*22
        #a = [2.5 for _ in range(11)]
        #b = [25 for _ in range(22)]
        #ub = a + b  # 决策变量上界
        lbin = [0] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        print(Vars)
        q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11 = Vars[:,[0]],Vars[:,[1]],Vars[:,[2]],Vars[:,[3]],Vars[:,[4]],Vars[:,[5]],Vars[:,[6]],Vars[:,[7]],Vars[:,[8]],Vars[:,[9]],Vars[:,[10]]
        
        # 相关参数
        beta = 0.5
        Ai = 3.4
        mi = 0.5
        l = [10.23, 1.8, 4.2, 5.8, 1.25, 1.2, 0.88, 1.03, 1.4, 1.9, 6.17, 1.8] # 下级渠道长度
        W = 280500  # 预计来水量
        S = [46, 146, 248, 65, 76, 55, 41, 117, 230, 233, 53]  # 下级渠道面积
        #q = [0.6, 1.0, 1.0, 0.6, 0.6, 0.5, 0.5, 0.6, 0.8, 1.5, 0.8]  # 下级渠道设计流量
        M = [1200 * i for i in S]  # 综合灌水定额
        Q = 2.5  # 上级渠道设计流量
        SQ = 1259  # 上级渠道面积

        T = 25

        # 目标函数
        # todo 针对每一个个体，最后再合并
        '''Quj = []
        for t in range(T):
            temp = 0
            for i in range(11):
                q, a, b = abs(Vars[:,[i]]),abs(Vars[:, [i + 11]]), abs(Vars[:, [i + 22]])
                #if (a.all() <= t.all() <= b.all()) or (b <= t <= a):
                if (a <= t <= b):
                    temp += q
            Quj.append(temp)
        Qu_bar = sum(Quj) / T
        f1 = sum([(val-Qu_bar)**2 for val in Quj]) / (T - 1)'''
        f1 = []
        for j in range(len(Vars)): # 种群大小
            Quj = []
            for t in range(T):
                temp = 0
                for i in range(11):
                    q, a, b = abs(Vars[j, [i]]),abs(Vars[j, [i + 11]]), abs(Vars[j, [i + 22]])
                    if (a <= t <= b) or (b <= t <= a):
                        temp += q
                Quj.append(temp)
            Qu_bar = sum(Quj) / T
            f1.append(sum([(val-Qu_bar)**2 for val in Quj]) / (T - 1))
        f1 = np.array(f1).reshape((len(Vars),1))
        print(f1)
        
        '''popsize = len(Vars[:,[0]])
        Quj = []
        f1 = []
        qt = 0
        for i in range(popsize):
            for t in range(1,25):
                for k in range(11):
                    q,tstart,tend = abs(Vars[i][k]),abs(Vars[i][k+11]),abs(Vars[i][k+22])
                    if(tstart<= t <=tend):
                        qt += q
                Quj.append(qt)
            print(Quj)
        Qu_bar = sum(Quj) / T
        f1 = sum([(val-Qu_bar)**2 for val in Quj]) / (T - 1)
        f1 = np.array(f1)'''
                
            

        temp_f2 = 0
        for i in range(11):
            qf2 = abs(Vars[:,[i]])
            period = abs(Vars[:, [i + 11]] - Vars[:, [i + 22]])
            temp_f2 += (beta * Ai * l[i] * (qf2 ** (1 - mi)) * period) / 100
        f2 = temp_f2
        '''temp_f2 = 0
        f2 = []
        for i in range(popsize):
            for j in range(11):
                qf2 = abs(Vars[i][j])
                period = abs(Vars[i][k+11]) - abs(Vars[i][k+22])
                temp_f2 += (beta * Ai * l[i] * (qf2 ** (1 - mi)) * period) / 100
        f2 = temp_f2
        f2 = np.array(f2)'''

        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV

        # ----------约束条件----------
        # 约束6
        constrain_6 = []
        qy6 = 0
        for i in range(11):
            qy6 = abs(Vars[:,[i]])
            period = abs(Vars[:, [i + 11]] - Vars[:, [i + 22]])
            constrain_6.append(qy6 * period)
        CV1 = (sum(constrain_6))*3600*24 - W

        # 约束7
        # todo  Mi是下级渠道控制区域下作物的灌溉定额
        CV2_0 = M * S[0] - q1 * abs(Vars[:, [0 + 11]] - Vars[:, [0 + 22]])*3600*24
        CV2_1 = M * S[1] - q2 * abs(Vars[:, [1 + 11]] - Vars[:, [1 + 22]])*3600*24
        CV2_2 = M * S[2] - q3 * abs(Vars[:, [2 + 11]] - Vars[:, [2 + 22]])*3600*24
        CV2_3 = M * S[3] - q4 * abs(Vars[:, [3 + 11]] - Vars[:, [3 + 22]])*3600*24
        CV2_4 = M * S[4] - q5 * abs(Vars[:, [4 + 11]] - Vars[:, [4 + 22]])*3600*24
        CV2_5 = M * S[5] - q6 * abs(Vars[:, [5 + 11]] - Vars[:, [5 + 22]])*3600*24
        CV2_6 = M * S[6] - q7 * abs(Vars[:, [6 + 11]] - Vars[:, [6 + 22]])*3600*24
        CV2_7 = M * S[7] - q8 * abs(Vars[:, [7 + 11]] - Vars[:, [7 + 22]])*3600*24
        CV2_8 = M * S[8] - q9 * abs(Vars[:, [8 + 11]] - Vars[:, [8 + 22]])*3600*24
        CV2_9 = M * S[9] - q10 * abs(Vars[:, [9 + 11]] - Vars[:, [9 + 22]])*3600*24
        CV2_10 = M * S[10] - q11 * abs(Vars[:, [10 + 11]] - Vars[:, [10 + 22]])*3600*24
        
        #结束时间大于开始时间约束
        

        pop.CV = np.hstack([CV1, CV2_0, CV2_1, CV2_2, CV2_3, CV2_4, CV2_5, CV2_6, CV2_7, CV2_8, CV2_9, CV2_10])
problem = MyProblem()  # 生成问题对象
"""==================================种群设置==============================="""
Encoding = 'BG'  # 编码方式
NIND = 100  # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器

population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
"""================================算法参数设置============================="""
myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
myAlgorithm.mutOper.Pm = 0.1  # 修改变异算子的变异概率
myAlgorithm.recOper.XOVR = 0.7  # 修改交叉算子的交叉概率
myAlgorithm.MAXGEN = 100   # 最大进化代数
myAlgorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = True  # 设置是否打印输出日志信息
myAlgorithm.drawing = 2  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
"""==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
[NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
NDSet.save()  # 把非支配种群的信息保存到文件中
"""==================================输出结果=============================="""
print('用时：%s 秒' % myAlgorithm.passTime)
print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
if NDSet.sizes != 0:
        print('最短完工时间为：%s' % NDSet.ObjV[0][0])
        print('最优调度方案为：')
        for i in range(NDSet.Phen.shape[1]):
            print(NDSet.Phen[0, i],end=" ")
else:
        print('没找到可行解。') 