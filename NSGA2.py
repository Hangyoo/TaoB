from nsga_ii import function1, function2
import numpy as np
import numpy as np
import geatpy as ea

"""
最小化目标双目标优化问题
min f1 
min f2 

s.t.
0.01<= x1 <= 0.3
0.01<= x2 <= 0.3
0.01<= x3 <= 0.3
0.01<= x4 <= 0.8
0.01<= x5 <= 0.4
0.01<= x6 <= 0.4
0.01<= x7 <= 0.25
0.01<= x8 <= 0.25
"""


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 8  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0.01, 0.01, 0.01,0.01, 0.01, 0.01,0.01, 0.01]  # 决策变量下界
        ub = [0.3,0.3,0.3,0.8,0.4,0.4,0.25,0.25]  # 决策变量上界
        lbin = [1,1,1,1,1,1,1,1]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1,1,1,1,1,1,1,1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)
        popsize = Vars.shape[0]
        F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
        F2 = np.array([float("Inf")] * popsize).reshape(popsize, 1)

        # 对目标中每个个体进行计算
        for i in range(popsize):
            # 读取变量取值
            x1 = Vars[i, [0]][0]  # a1
            x2 = Vars[i, [1]][0]  # a2
            x3 = Vars[i, [2]][0]  # a3
            x4 = Vars[i, [3]][0]  # a4
            x5 = Vars[i:, [4]][0] # b1
            x6 = Vars[i, [5]][0]  # b2
            x7 = Vars[i, [6]][0]  # c1
            x8 = Vars[i, [7]][0]  # c2

            # 计算目标函数
            f1 = function1(x1,x2,x3,x4,x5,x6,x7,x8)
            f2 = function2(x1,x2,x3,x4,x5,x6,x7,x8)

            F1[i, 0] = f1
            F2[i, 0] = f2

        pop.ObjV = np.hstack([F1, F2])  # 把求得的目标函数值赋值给种群pop的ObjV


if __name__ == '__main__':
    """===============================实例化问题对象============================"""
    problem = MyProblem()  # 生成问题对象
    """==================================种群设置==============================="""
    Encoding = 'BG'  # 编码方式
    NIND = 100  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器

    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """================================算法参数设置============================="""
    myAlgorithm = ea.moea_NSGA2_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.mutOper.Pm = 0.1  # 修改变异算子的变异概率
    myAlgorithm.recOper.XOVR = 0.8  # 修改交叉算子的交叉概率
    myAlgorithm.MAXGEN = 50  # 最大进化代数
    myAlgorithm.logTras = 0  # 设置每多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = False  # 设置是否打印输出日志信息
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