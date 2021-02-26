
import numpy as np
import geatpy as ea
 
class MyProblem(ea.Problem): # 继承Problem父类
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1
        maxormins = [1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 234
        varTypes = [1] * 234
        lb = [0] * 234
        ub = [8] * 234
        lbin = [1] * 234
        ubin = [1] * 234
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        time = [138, 47, 132, 98, 48, 161, 65, 102, 98, 40, 89, 94, 155, 75, 281, 36, 267, 47, 33, 264, 333, 49, 171, 51,
                11, 76]

        popsize = len(Vars[:, [0]])
        each_time = [[0]*9 for _ in range(popsize)]  # 记录9组各自的完工时间
        for i in range(popsize):
            for j in range(len(Vars[0, :])):
                var = int(Vars[i, [j]])
                each_time[i][var] += time[var]

        # 目标函数最大的最小
        f = []
        for i in range(popsize):
            f.append([max(each_time[i])])
        f = np.array(f)
        pop.ObjV = f  # 把求得的目标函数值赋值给种群pop的ObjV



"""===============================实例化问题对象================================"""
problem = MyProblem() # 生成问题对象
"""==================================种群设置=================================="""
Encoding = 'BG'       # 编码方式
NIND = 100            # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
"""================================算法参数设置================================="""
myAlgorithm = ea.soea_EGA_templet(problem, population) # 实例化一个算法模板对象

myAlgorithm.MAXGEN = 300 # 最大进化代数
myAlgorithm.logTras = 50  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
myAlgorithm.verbose = True  # 设置是否打印输出日志信息
myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
"""===========================调用算法模板进行种群进化==============--==========="""
[BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
BestIndi.save()  # 把最优个体的信息保存到文件中
"""==================================输出结果=================================="""
print('用时：%f 秒' % myAlgorithm.passTime)
if BestIndi.sizes != 0:
    print('最短完工时间为：%s' % BestIndi.ObjV[0][0])
    print('最优调度方案为：')
    for i in range(BestIndi.Phen.shape[1]):
        print(BestIndi.Phen[0, i],end=" ")
else:
    print('没找到可行解。')

