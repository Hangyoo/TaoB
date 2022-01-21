import geatpy as ea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings

warnings.filterwarnings('ignore')

"""
    主函数：通过调用机器学习模型，用于评估进化算法的适应度值

    决策变量的最大化单目标优化问题
    max f1
    
    Depth 500 - 7655
    x1:钻头尺寸	243.3 - 333.4
    钻头类型(忽略) 0 - 3
    岩性(忽略)	0 - 7
    x2:钻压      0 - 324
    x3:钻速	    0 - 232
    x4:流量	    0 -3944.4
    x5:立管压力	0.2 - 27.8
    x6:钻井液密度	1.0 - 2.15
    x7:声波系数	42.934 - 111.708
    x8:伽马系数	0.970 - 153.351
"""

# todo 设置当前的深度
Depth = 7500

Val1 = 1.0 # 默认钻头为牙轮
Val2 = 0.142857 # 默认岩性为泥岩
max_val1,min_val1 = 3, 0
max_val2,min_val2 = 7, 0
max_depth, min_depth = 7655, 500


# 读取训练好的机器学习模型
with open("RF.pkl", "rb") as f1:  # 预测loss
    model1 = pickle.load(f1)


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=1):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 8  # 初始化Dim（决策变量维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0]*Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [243.3, 0, 0, 0, 0.2, 1.0, 42.934, 0.970]  # 决策变量下界
        ub = [333.4,324,232,3944.4,27.8,2.15,111.708,153.351]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("-Inf")] * popsize).reshape(popsize, 1)
        for i in range(popsize):
            # 归一化到0-1之间
            # depth = (Depth - min_depth)/(max_depth-min_depth)
            # val1 = (Val1 - min_val1)/(max_val1-min_val1)
            # val2 = (Val2 - min_val2)/(max_val2-min_val2)
            # x1 = (Vars[i, [0]][0] - lb[0]) / (ub[0]-lb[0])
            # x2 = (Vars[i, [1]][0] - lb[1]) / (ub[1]-lb[1])
            # x3 = (Vars[i, [2]][0] - lb[2]) / (ub[2]-lb[2])
            # x4 = (Vars[i, [3]][0] - lb[3]) / (ub[3]-lb[3])
            # x5 = (Vars[i, [4]][0] - lb[4]) / (ub[4]-lb[4])
            # x6 = (Vars[i, [5]][0] - lb[5]) / (ub[5]-lb[5])
            # x7 = (Vars[i, [6]][0] - lb[6]) / (ub[6]-lb[6])
            # x8 = (Vars[i, [7]][0] - lb[7]) / (ub[7]-lb[7])
            x1 = Vars[i, [0]][0]
            x2 = Vars[i, [1]][0]
            x3 = Vars[i, [2]][0]
            x4 = Vars[i, [3]][0]
            x5 = Vars[i, [4]][0]
            x6 = Vars[i, [5]][0]
            x7 = Vars[i, [6]][0]
            x8 = Vars[i, [7]][0]

            X = np.array([Depth, x1, Val1, Val2, x2, x3, x4, x5, x6,x7,x8]).reshape(1, 11)
            # 计算目标函数值
            f1 = model1.predict(X)  # loss
            F1[i, 0] = f1
        pop.ObjV = np.hstack([F1])  # 把求得的目标函数值赋值给种群pop的ObjV

if __name__ == '__main__':
    """===============================实例化问题对象============================"""
    problem = MyProblem()     # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'           # 编码方式
    NIND = 50               # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))    # 创建区域描述器

    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    #myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm = ea.soea_SGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.1    # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.8  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 30        # 最大进化代数
    myAlgorithm.logTras = 10         # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False     # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0        # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化=========================
    调用run执行算法模板，得到帕累托最优解集NDSet以及最后一代种群。NDSet是一个种群类Population的对象。
    NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
    详见Population.py中关于种群类的定义。
    """
    [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
    # 对目标进行筛选
    Variable = []
    for i in range(len(NDSet.Phen.tolist())):
        v = NDSet.Phen.tolist()[i]
        Variable.append(v)
    NDSet.Phen = np.array(Variable)
    NDSet.sizes = len(Variable)
    problem.aimFunc(NDSet)
    df1 = pd.DataFrame(NDSet.Phen)
    df2 = pd.DataFrame(NDSet.ObjV)
    df1.to_csv('Variable.csv',header=None, index=None)
    df2.to_csv('Objective.csv',header=None, index=None)
    # 结果输出
    print(f'{Depth}深度下最快速度为:{NDSet.ObjV[0][0]}')
    # 变量输出
    print(f'钻头尺寸:{Variable[0][0]}')
    print(f'钻压:{Variable[0][1]}')
    print(f'钻速:{Variable[0][2]}')
    print(f'流量:{Variable[0][3]}')
    print(f'立管压力:{Variable[0][4]}')
    print(f'钻井液密度:{Variable[0][5]}')
    print(f'声波系数:{Variable[0][6]}')
    print(f'伽马系数:{Variable[0][7]}')

    # NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % myAlgorithm.passTime)
