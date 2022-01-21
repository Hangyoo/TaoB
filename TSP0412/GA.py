from TSP0412.Individual import *
from TSP import *
from TSP0412.Population import *
import numpy as np

# crossover rate
c_rate = 0.8
# mutate rate
m_rate = 0.2

indiv = None


def cross(p1, p2):
    """
    if np.random.rand() > c_rate:
        return p2

    index1 = np.random.randint(0, len(p1) - 1)
    index2 = np.random.randint(index1, len(p1) - 1)
    part = p2[index1:index2]
    new = []
    length = 0
    for g in p1:
        if length == index1:
            new.extend(part)
        if g not in part:
            new.append(g)
        length = length + 1

    new = np.array(new)
    return new

    """

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
    """
    if np.random.rand() > m_rate:
        return p
    copy = []
    for i in p:
        copy.append(i)
    # find 2 random cities
    begin, end = sorted(random.sample(range(len(copy)), 2))
    copy[begin], copy[end] = copy[end], copy[begin]
    copy = np.array(copy)
    return copy
    """
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
