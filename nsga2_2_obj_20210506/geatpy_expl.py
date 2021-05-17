import numpy as np
import geatpy as ea


#继承Problem父类
class MyProblem(ea.Problem):

       def __init__(self, x1up, x2up, x3up, x4up, x5up, x6up, x7up, x8up):
              vname ='BNH'# 初始化name（函数名称，可以随意设置）
              M = 2 # 初始化M（目标维数）
              maxormins = [-1] * M # 初始化maxormins
              Dim = 11 # 初始化Dim（决策变量维数）
              varTypes = [0] * Dim # 初始化varTypes（决策变量的类型，0：实数；1：整数）
              lb = [0]*Dim # 决策变量下界
              ub = [x1up, x2up, x3up, x4up, x5up, x6up, x7up, x8up, 80000, 80000, 15000]# 决策变量上界
              lbin = [1] * Dim  # 决策变量下边界
              ubin = [1] * Dim # 决策变量上边界
              # 调用父类构造方法完成实例化
              ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb,ub, lbin, ubin)


       def aimFunc(self, pop):
              # 目标函数
              Vars = pop.Phen
              # 得到决策变量矩阵
              x1 = Vars[:, [0]]
              x2 = Vars[:, [1]]
              x3 = Vars[:, [2]]
              x4 = Vars[:, [3]]
              x5 = Vars[:, [4]]
              x6 = Vars[:, [5]]
              x7 = Vars[:, [6]]
              x8 = Vars[:, [7]]
              x9 = Vars[:, [8]]
              x10 = Vars[:, [9]]
              x11 = Vars[:, [10]]
              f1 = x1+x2+x3-x4-x5-x6-x7-x8-x9-x10-x11
              f2 = 0.8*x8 + 1000
              # 采用可行性法则处理约束可以自己添加新的约束
              '''pop.CV = np.hstack(
              [(x1 - 5)**2 + x2**2 - 25,
              -(x1 - 8)**2 - (x2 - 3)**2 + 7.7])'''
              # 把求得的目标函数值赋值给种群pop的ObjV
              pop.ObjV = np.hstack([f1, f2])

       def calReferObjV(self): # 计算全局最优解
              N = 10000 # 欲得到10000个真实前沿点
              x1 = np.linspace(0, 5, N)
              x2 = x1.copy()
              x2[x1 >= 3] = 3
              return np.vstack((4 * x1**2 + 4 * x2**2,
              (x1 - 5)**2 + (x2 - 5)**2)).T


"""
执行脚本
"""
if __name__ == "__main__":
       np.set_printoptions(suppress=True)
       np.set_printoptions(threshold=np.inf)
       x1up = input('请输入产生单元x1的上限：')
       x2up = input('请输入产生单元x2的上限：')
       x3up = input('请输入产生单元x3的上限：')
       x4up = input('请输入消耗单元x4的上限：')
       x5up = input('请输入消耗单元x5的上限：')
       x6up = input('请输入调节单元x6的上限：')
       x7up = input('请输入调节单元x7的上限：')
       x8up = input('请输入调节单元x8的上限：')
       """=========================实例化问题对象==========================="""
       problem = MyProblem(x1up, x2up, x3up, x4up, x5up, x6up, x7up, x8up) # 实例化问题对象
       """===========================种群设置=============================="""
       Encoding = 'RI' # 编码方式
       NIND = 100 # 种群规模
       Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges,
       problem.borders) # 创建区域描述器
       population = ea.Population(Encoding, Field, NIND) #实例化种群对象（此时种群还没被真正初始化，仅仅是生成一个种群对象）
       """=========================算法参数设置============================"""
       myAlgorithm = ea.moea_NSGA2_templet(problem, population) #实例化一个算法模板对象
       myAlgorithm.MAXGEN = 200 # 最大遗传代数
       myAlgorithm.drawing = 1 # 设置绘图方式
       """===================调用算法模板进行种群进化=======================
       调用run执行算法模板，得到帕累托最优解集NDSet。
       NDSet是一个种群类Population的对象。
       NDSet.ObjV为最优解个体的目标函数值；NDSet.Phen为对应的决策变量值。
       详见Population.py中关于种群类的定义。
       """
       [NDSet, population] = myAlgorithm.run()  # 执行算法模板，得到非支配种群以及最后一代种群
       temp = 0
       j = 0
       resultmap = []
       for i in NDSet.ObjV:
              if j == 0:
                     temp = i[0]
                     j = j + 1
                     continue
              if i[0] < temp:
                     temp = i[0]
                     k = j
              j = j + 1
              if j == len(NDSet.ObjV):
                     resultmap.append(temp)
                     resultmap.append(k)

       print('结果'+ str(NDSet.ObjV[resultmap[1]]))
       print('变量取值'+ str(NDSet.Phen[resultmap[1]]))
       NDSet.save()  #把非支配种群的信息保存到文件中
       """==================================输出结果=============================="""
       print('用时：%s 秒' % myAlgorithm.passTime)
       print('非支配个体数：%d 个' % NDSet.sizes) if NDSet.sizes != 0 else print('没有找到可行解！')
       