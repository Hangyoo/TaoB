import copy
import random
import itertools
from taobao import config


# 4.3.2 交叉操作
###########################

#  POX交叉
def precedenceOperationCrossover(p1, p2, parameters):
    J = parameters['jobs']
    jobNumber = len(J)
    jobsRange = range(1, jobNumber+1)
    sizeJobset1 = random.randint(0, jobNumber)
    # 从工件集jobsRange中随机获取sizeJobset1个元素，作为一个列表返回
    jobset1 = random.sample(jobsRange, sizeJobset1)

    o1 = []   #子代1
    p1kept = []
    for i in range(len(p1)):
        e = p1[i]         #获取基因
        if e in jobset1:  #若属于集合J1
            o1.append(e)  #加入o1中
        else:
            o1.append(-1) #加入-1
            p1kept.append(e)   #在p1kept中加入e

    o2 = []   #子代2
    p2kept = []
    for i in range(len(p2)):
        e = p2[i]
        if e in jobset1:
            o2.append(e)
        else:
            o2.append(-1)
            p2kept.append(e)

    for i in range(len(o1)):
        if o1[i] == -1:
            o1[i] = p2kept.pop(0)  #若o1对应基因为-1，则用p2kept[0]替换

    for i in range(len(o2)):
        if o2[i] == -1:
            o2[i] = p1kept.pop(0)
    return (o1, o2)

#  两点交叉
def twoPointCrossover(p1, p2):
    pos1 = random.randint(0, len(p1) - 1)
    pos2 = random.randint(0, len(p2) - 1)
    if pos1 > pos2:    #生成两点，并使 pos1 < pos2
        pos2, pos1 = pos1, pos2

    Child1 = p1
    if pos1 != pos2:
        Child1 = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]

    Child2 = p2
    if pos1 != pos2:
        Child2 = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]
    return (Child1, Child2)

#基于工序编码部分交叉（采用POX方式）
def crossoverOS(p1, p2, parameters):
    return precedenceOperationCrossover(p1, p2, parameters)

#基于机器安排部分交叉(采用两点交叉）
def crossoverMS(p1, p2):
    return twoPointCrossover(p1, p2)

#最终交叉操作
def crossover(population, parameters):
    newPop = []   #新种群
    i = 0
    while i < len(population):  #依次对种群中染色体执行交叉操作
        # 种群 ((OS, MS)，(OS, MS)，(OS, MS))
        (OS1, MS1, WS1) = population[i]
        (OS2, MS2, WS2) = population[i+1]

        if random.random() < config.pc:
            (OS1_Child, OS2_Child) = crossoverOS(OS1, OS2, parameters)
            (MS1_Child, MS2_Child) = crossoverMS(MS1, MS2)
            newPop.append((OS1_Child, MS1_Child, WS1))
            newPop.append((OS2_Child, MS2_Child, WS2))
        else:  #满足则交叉，否则不变
            newPop.append((OS1, MS1, WS1))
            newPop.append((OS2, MS2, WS2))
        i = i + 2
    return newPop


# 4.3.3 变异操作（返回变异后种群）
##########################

#1 互换变异OS
def swappingMutation(p):
    Child = copy.copy(p)   #生成两点，互换两点基因
    pos1 = random.randint(0, len(p) - 1)
    pos2 = random.randint(0, len(p) - 1)
    if pos1 == pos2:
        return p
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    Child[pos2], Child[pos1] = Child[pos1], Child[pos2]
    return Child

#基于机器编码变异（选择一半的基因，进行机器变异）MS
def halfMutation(p, parameters):
    o = p
    jobs = parameters['jobs']  #获取工件数
    length = len(p)  #染色体长度
    r = int(length/2)
    positions = random.sample(range(length), r)  #在length中生成个r个数，返回列表

    i = 0
    for job in jobs:
        for op in job:
            if i in positions:
                o[i] = random.randint(0, len(op)-1)
            i = i+1
    return o

#工序变异采用1，2 两种中随机一种
def mutationOS(p):
    return swappingMutation(p)

def mutationMS(p, parameters):
    return halfMutation(p, parameters)

#最终变异
def mutation(population, parameters):
    newPop = []
    for (OS, MS, WS) in population:
        if random.random() < config.pm:
            OS_Child = mutationOS(OS)
            MS_Child = mutationMS(MS, parameters)
            newPop.append((OS_Child, MS_Child, WS))
        else:
            newPop.append((OS, MS, WS))
    return newPop
