from TSP0417.Individual import *
from TSP0417.TSP import *
from TSP0417.Population import *
import numpy as np

# crossover rate
c_rate = 0.8
# mutate rate
m_rate = 0.2

indiv = None


def cross(p1, p2):

    # Partial-Mapped Crossover (PMX)
    if np.random.rand() > c_rate:
        return p2

    size = len(p2)
    copy1 = []
    copy2 = []
    for i in p1:
        copy1.append(i)
    for i in p2:
        copy2.append(i)
    x1, x2 = [0] * size, [0] * size
    for i in range(size):
        x1[copy1[i]] = i
        x2[copy2[i]] = i

    index1 = np.random.randint(0, size)
    index2 = np.random.randint(0, size - 1)
    if index2 >= index1:
        index2 += 1
    else:
        index1, index2 = index2, index1
    for i in range(index1, index2):
        temp1 = copy1[i]
        temp2 = copy2[i]

        copy1[i], copy1[x1[temp2]] = temp2, temp1
        copy2[i], copy2[x2[temp1]] = temp1, temp2

        x1[temp1], x1[temp2] = x1[temp2], x1[temp1]
        x2[temp1], x2[temp2] = x2[temp2], x2[temp1]

    copy1 = np.array(copy1)
    return copy1


def mutate(p):
    if np.random.rand() > m_rate:
        return p
    index1 = np.random.randint(0, len(p) - 1)
    index2 = np.random.randint(index1, len(p) - 1)

    if index1 == index2:
        return p
    part = p[index1:index2]
    new = []
    length = 0
    for g in p:
        if length == index1:
            new.extend(part[::-1])
        if g not in part:
            new.append(g)
        length = length + 1
    new = np.array(new)
    return new

def mutate_improve(p):
    '''原变异方式中若两个索引位置一样，则不执行变异。但这个方式的缺点是在满足变异条件的情况下也不能执行变异操作，会使得种群的多样性不足。
       改进1：若满足变异条件，则会执行变异操作。若两点位置不一样，则重新生成两位置.
       改进2：使得index1  < index2'''
    if np.random.rand() > m_rate:
        return p
    index1 = np.random.randint(0, len(p) - 1)
    index2 = np.random.randint(0, len(p) - 1)
    while index1 == index2:
        index2 = np.random.randint(0, len(p) - 1)

    if index1 > index2:
        index1, index2 = index2, index1

    part = p[index1:index2]
    new = []
    length = 0
    for g in p:
        if length == index1:
            new.extend(part[::-1])
        if g not in part:
            new.append(g)
        length = length + 1
    new = np.array(new)
    return new



def select(pop, tsp):
    new = []

    for p in pop:
        if p.rank <= 2:
            new.append(p)
    while len(new) != len(pop):
        new_i = Individual()
        new_i = new_i.create_indiv(tsp)
        new.append(new_i)

    return new
