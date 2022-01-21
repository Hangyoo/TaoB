# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

'''
经典PSO算法
max 单目标, 2维度连续问题
'''

def getweight():
    # 惯性权重
    weight = 1
    return weight

def getlearningrate():
    # 分别是粒子的 个体学习因子 和 社会的学习因子，(也称为加速常数)
    lr = (0.49445,1.49445)
    return lr

def getmaxgen():
    # 最大迭代次数
    maxgen = 300
    return maxgen

def getsizepop():
    # 种群规模
    sizepop = 50
    return sizepop

def getrangepop():
    # 粒子的位置的范围限制,x、y方向的限制相同
    rangepop = (-2*math.pi , 2*math.pi)
    #rangepop = (-2,2)
    return rangepop

def getrangespeed():
    # 粒子的速度范围限制
    rangespeed = (-0.5,0.5)
    return rangespeed

def func(x):
    # 输入: x 粒子位置
    # 输出: y 粒子适应度值
    if (x[0]==0) & (x[1]==0):
        y = np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    else:
        y = np.sin(np.sqrt(x[0]**2+x[1]**2))/np.sqrt(x[0]**2+x[1]**2)+np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    return y

def initpopvfit(sizepop):
    pop = np.zeros((sizepop,2)) # 种群中每个粒子的位置
    v = np.zeros((sizepop,2))   # 种群中每个粒子的速度
    fitness = np.zeros(sizepop) # 种群中每个粒子的适应度值函数

    for i in range(sizepop):
        # 初始化种群和速度
        pop[i] = [(np.random.rand()-0.5)*rangepop[0]*2,(np.random.rand()-0.5)*rangepop[1]*2]
        v[i] = [(np.random.rand()-0.5)*rangepop[0]*2,(np.random.rand()-0.5)*rangepop[1]*2]
        fitness[i] = func(pop[i])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体中最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    # 每个粒子最优位置及其适应度值 (使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似)
    pbestpop,pbestfitness = pop.copy(),fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness  

w = getweight()          # 惯性权重
lr = getlearningrate()   # 个体学习因子 和 社会的学习因子
maxgen = getmaxgen()     # 最大迭代次数
sizepop = getsizepop()   # 种群规模
rangepop = getrangepop() # 粒子的位置的范围限制,x、y方向的限制相同
rangespeed = getrangespeed() # 粒子的速度的范围限制

pop,v,fitness = initpopvfit(sizepop) # 初始化
gbestpop,gbestfitness,pbestpop,pbestfitness = getinitbest(fitness,pop) # 结果记录

result = np.zeros(maxgen) # 记录中间迭代结果

for i in range(maxgen):
        t=0.5
        #速度更新
        for j in range(sizepop):
            v[j] += lr[0]*np.random.rand()*(pbestpop[j]-pop[j])+lr[1]*np.random.rand()*(gbestpop-pop[j])
        # 越界修复
        v[v<rangespeed[0]] = rangespeed[0]
        v[v>rangespeed[1]] = rangespeed[1]

        #粒子位置更新
        for j in range(sizepop):
            #pop[j] += 0.5*v[j]
            pop[j] = t*(0.5*v[j])+(1-t)*pop[j]
        # 越界修复
        pop[pop<rangepop[0]] = rangepop[0]
        pop[pop>rangepop[1]] = rangepop[1]

        # 适应度更新
        for j in range(sizepop):
            fitness[j] = func(pop[j])

        # 个人最好适应度值更新
        for j in range(sizepop):
            if fitness[j] > pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()

        # 全局最好适应度值更新
        if pbestfitness.max() > gbestfitness :
            gbestfitness = pbestfitness.max()
            gbestpop = pop[pbestfitness.argmax()].copy()

        result[i] = gbestfitness


plt.plot(result)
plt.show()