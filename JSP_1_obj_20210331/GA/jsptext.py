import numpy as np
import random
from JSP_1_obj_20210331.GA.caltime import cal
from JSP_1_obj_20210331.GA.evolve import mutation,across,select
import matplotlib.pyplot as plt
from JSP_1_obj_20210331.GA.generator_chrom import generatechrom

def creatchrom(ChromChild,ChromPar,NIND,T):   #取个体前25%，父代前50%，随机生成25%
    a = int(1* NIND)
    Chrom1 = ChromChild[1:a+1,:]
    b = int(1 * NIND)
    Chrom2 = ChromPar[1:b+1,:]
    Chrom3 = generatechrom(T,(int(2*NIND)-a-b))
    Chrom = np.concatenate((Chrom1,Chrom2,Chrom3))
    return Chrom


def JSP_NSGA(T,Jm,NIND,MAXGEN,Pc,Pm,GGAP):
    #比如 JSP(T, Jm, 40, 500, 0.9, 0.8, 0.6)

    # NIND = 60;    % 个体数目(Number of individuals)
    # MAXGEN = 500; % 最大遗传代数(Maximum number of generations)
    # GGAP = 0.9;   % 代沟(Generation gap)
    # XOVR = 0.8;   % 交叉率
    # MUTR = 0.6;   % 变异率

    PNumber, MNumber = T.shape
    gen = 0   #记录迭代次数
    trace = np.zeros((2,MAXGEN))   #寻优结果的初始值.

    WNumber = PNumber*MNumber
    Chrom = np.zeros((NIND,WNumber))

    # 产生初始种群，以及初始种群中个体的编码
    Number = [0] * PNumber

    for j in range(NIND):

        for i in range(PNumber):
            Number[i] = MNumber  # Number总储存了每个工件的工序数
        Numbertemp = Number

        for i in range(WNumber):

            val = random.randint(1,PNumber)   #随机选一个工作号，从1开始

            while Numbertemp[val-1] == 0:       #如果工作对应的工序数不为0，那么可以安排工作
                val = random.randint(1, PNumber)

            Chrom[j,i] = val
            Numbertemp[val-1] -= 1               #安排一个工序后，就可以给这个工件剩余工序-1


    #计算目标函数值
    PVal, P, objV, temp = cal(Chrom,T,Jm)    #temp 为2*NIND ndarry数组。 个体---适应值（完工时间）

    while gen < MAXGEN:

        print('剩余迭代%d次'%(MAXGEN-gen))
        if gen == 0:
            fronts,Chromchild = fast_nondominated_sort(Chrom,NIND,T,Jm)
        else:
            Chrom_Sel,Chromelite = select(Chromchild,temp,GGAP)
            Chrom_Acr = across(Chrom_Sel,Pc,PNumber)
            Chrom_MUT = mutation(Chrom_Acr,Pm)    #Chrom 中的个体数量为 GGAP*NIND
            ChromPar = np.concatenate((Chromelite,Chrom_MUT))    #将父代子代合并了，种群数量为2*NIND
            Chrom = np.concatenate((Chromchild,ChromPar))
            #Chrom = creatchrom(Chromchild, ChromPar, NIND, T)
            fronts, Chromchild = fast_nondominated_sort(Chrom,NIND,T, Jm)    #选出前NIND个个体，组成Chrom 返回

        #计算目标函数的值
        PVal, P, objV, temp = cal(Chrom, T, Jm)

        trace[0,gen] = min(objV)
        trace[1,gen] = sum(objV)/len(objV)

        #输出最优解及其序号
        if gen == 0 :
            val1 = PVal
            val2 = P
            MinVal = min(objV)

        else:
            if MinVal > trace[0,gen]:
                val1 = PVal
                val2 = P
                MinVal = trace[0,gen]

        gen += 1

    PVal = val1
    P = val2

    print('最优的完工时间为：',MinVal)
    print(P)
    print(PVal)

    return trace

def plot(trace):
    minval = []
    meanval = []
    x = []

    for i in range(trace.shape[1]):
        minval.append(trace[0, i])
        meanval.append(trace[1, i])
        x.append(i)

    plt.plot(x,minval,'r',x,meanval,'temp1')
    plt.title('每代平均适应度值', fontproperties='SimHei', fontsize=10)
    plt.xlabel('迭代次数', fontproperties='SimHei', fontsize=10)
    plt.ylabel('适应度值', fontproperties='SimHei', fontsize=10)
    plt.show()



