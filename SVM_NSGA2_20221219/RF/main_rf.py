import geatpy as ea
import numpy as np
import pandas as pd
from SVM_NSGA2_20221219.RF.problem import MyProblem
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

"""
    主函数：通过调用机器学习模型，用于评估进化算法的适应度值
"""

if __name__ == '__main__':
    """===============================实例化问题对象============================"""
    problem = MyProblem()     # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'RI'           # 编码方式
    NIND = 100               # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders,
                      [10] * len(problem.varTypes))    # 创建区域描述器

    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    #myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.1    # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.8  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 100        # 最大进化代数
    myAlgorithm.logTras = 10         # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True     # 设置是否打印输出日志信息
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

     # 绘图
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    xd = [x[0] for x in NDSet.ObjV]
    yd = [x[1] for x in NDSet.ObjV]
    zd = [x[2] for x in NDSet.ObjV]
    ax1.scatter3D(xd,yd,zd, marker='o', s=15)  #绘制散点图
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    ax1.set_zlabel('f3')
    plt.show()

    # NDSet.save()  # 把非支配种群的信息保存到文件中
    """==================================输出结果=============================="""
    print('用时：%s 秒' % myAlgorithm.passTime)
    print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
