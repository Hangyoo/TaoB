import numpy as np
import random
from JSP_1_obj_20210331.GA.caltime import *
from JSP_1_obj_20210331.GA.evolve import mutation,across,select
import matplotlib.pyplot as plt
import time
from JSP_1_obj_20210331.GA.caltime import readJobs
from JSP_1_obj_20210331 import config
from JSP_1_obj_20210331.util import gantt


def JSP(T,Jm,NIND,MAXGEN,Pc,Pm,GGAP):
    #比如 JSP(T, Jm, 40, 500, 0.9, 0.8, 0.6)

    # NIND = 60;    % 个体数目(Number of individuals)
    # MAXGEN = 500; % 最大遗传代数(Maximum number of generations)
    # GGAP = 0.9;   % 代沟(Generation gap) 忽略不用看
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
        if gen % 10 == 0:
            print('剩余迭代次数：',MAXGEN-gen)
        Chrom_Sel,Chromelite = select(Chrom,temp,GGAP)
        Chrom_Acr = across(Chrom_Sel,Pc,PNumber)
        Chrom_MUT = mutation(Chrom_Acr,Pm)    #Chrom 中的个体数量为 GGAP*NIND
        Chrom = np.concatenate((Chromelite,Chrom_MUT))


        #计算目标函数的值
        PVal, P, objV, temp = cal(Chrom, T, Jm)
        temp = sorted(temp)
        gen_min = min(objV)
        trace[0,gen] = gen_min

        trace[1,gen] = sum(objV)/len(objV)

        #输出最优解及其序号
        if gen == 0 :
            val1 = PVal
            val2 = P
            MinVal = min(objV)

        else:
            if MinVal > gen_min:
                val1 = PVal
                val2 = P
                MinVal = gen_min

        gen += 1

    PVal = val1
    P = val2

    #解码P并返回最优调度工序
    best_individual = decode(P,T)

    print('最优的完工时间为：',MinVal)
    print('最优调度工序为:',best_individual)
    print('工序开完工时间为',PVal)

    # 解码甘特图信息
    gantt = {}
    mac = [[] for _ in range(MNumber)]
    nr= [0] * PNumber  # 记录工件加工到第几个工序
    for i in range(len(best_individual)):
        job = best_individual[i]
        macIdx = Jm[job-1][nr[job-1]]  # 工件号  工序号
        #print(i-1,nr[i-1],macIdx)
        mac[macIdx].append([PVal[0][i],PVal[1][i],str(job)+'-'+str(nr[job-1])])
        nr[job-1] += 1
    for i in range(MNumber):
        gantt['Machine'+str(i+1)] = mac[i]
    print(gantt)
    return trace,gantt

def plot(trace):
    minval = []
    meanval = []
    x = []

    for i in range(trace.shape[1]):
        minval.append(trace[0, i])
        meanval.append(trace[1, i])
        x.append(i)

    print(minval)
    print(meanval)

    plt.plot(x,minval,'r',x,meanval,'temp1')
    plt.title('每代平均适应度值', fontproperties='SimHei', fontsize=10)
    plt.xlabel('迭代次数', fontproperties='SimHei', fontsize=10)
    plt.ylabel('适应度值', fontproperties='SimHei', fontsize=10)
    plt.show()



if __name__ == '__main__':
    # Benchmark文件路径
    path = r'C:\Users\Hangyu\PycharmProjects\TaoB\JSP_1_obj_20210331\data\ft10.txt'
    Jm, T, makespan = readJobs(path)
    #todo 具体参数设置 去config文件内设置
    trace,gandata = JSP(T, Jm, config.N, config.GA_epoch, config.pc, config.pm, 0.9)
    gantt.draw_chart(gandata)
    # 结果保存
    save_rest(trace[0],r'C:\Users\DELL\PycharmProjects\TB\JSP_1_obj_20210331\result\gbegin' + path[-8:-4] + '.pkl')






