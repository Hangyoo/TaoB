import copy
import random
import itertools
from NSGA2 import config

###########################


#  两点交叉
def crossover_oper(p1, p2):
    # 对两个个体执行SBX交叉操作
    lower_bound, upper_bound = 0,11
    distribution_index = 20
    for j in range(44):
        # 对某自变量交叉
        if abs(p1[j]-p2[j]) > 1.0E-14:
            if p1[j] < p2[j]:
                y1,y2 = p1[j],p2[j]
            else:
                y2, y1 = p2[j], p1[j]

            beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
            alpha = 2.0 - pow(beta, -(distribution_index + 1.0))

            # rand = random.random()
            # if rand <= (1.0 / alpha):
            #     betaq = pow(rand * alpha, (1.0 / (distribution_index + 1.0)))
            # else:
            #     betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))

            betaq = random.random()

            c1 = int(0.5 * (y1 + y2 - betaq * (y2 - y1)))

            beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
            alpha = 2.0 - pow(beta, -(distribution_index + 1.0))

            # if rand <= (1.0 / alpha):
            #     betaq = pow((rand * alpha), (1.0 / (distribution_index + 1.0)))
            # else:
            #     betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (distribution_index + 1.0))

            betaq = random.random()

            c2 = int(0.5 * (y1 + y2 + betaq * (y2 - y1)))

            if c1 < lower_bound:
                c1 = lower_bound
            if c2 < lower_bound:
                c2 = lower_bound
            if c1 > upper_bound:
                c1 = upper_bound
            if c2 > upper_bound:
                c2 = upper_bound

            if random.random() <= 0.5:
                p1[j] = c2
                p2[j] = c1
            else:
                p1[j] = c1
                p2[j] = c2
        Child1 = p1
        Child2 = p2

    return Child1, Child2



#最终交叉操作
def crossover(population):
    newPop = []   #新种群
    i = 0
    while i < len(population):  #依次对种群中染色体执行交叉操作

        p1 = population[i]
        p2 = population[i+1]

        if random.random() < config.pc:
            Child1, Child2 = crossover_oper(p1,p2)
            newPop.append(Child1)
            newPop.append(Child2)
        else:  #满足则交叉，否则不变
            newPop.append(p1)
            newPop.append(p2)
        i = i + 2
    return newPop
    # return population


# 变异操作
def Mutation(p):
    Child = copy.copy(p)   #生成两点，互换两点基因
    pos1 = random.randint(0, len(p) - 1)
    pos2 = random.randint(0, len(p) - 1)
    if pos1 == pos2:
        return p
    if pos1 > pos2:
        pos1, pos2 = pos2, pos1
    Child[pos1] = random.randint(0,11)
    Child[pos2] = random.randint(0,11)
    return Child



#最终变异
def mutation(population):
    newPop = []
    for p in population:
        if random.random() < config.pm:
            Child = Mutation(p)
            newPop.append(Child)
        else:
            newPop.append(p)
    return newPop
