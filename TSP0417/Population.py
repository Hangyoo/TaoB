from TSP0417.Individual import *
import TSP0417.GA as ga
import copy as Copy
import numpy as np

class Population:
    # population size
    size = 100

    indiv = Individual()

    def __init__(self):
        pass

    def create_pop(self, tsp):
        P = []
        for i in range(self.size):
            indiv = self.indiv.create_indiv(tsp)
            P.append(indiv)

        return P


    def next_pop(self, P, tsp):
        next = []
        indiv = self.indiv
        pop = P
        size = len(pop)
        for i in pop:
            i = indiv.reset_indiv(i)
            copy = indiv.create_indiv(tsp)
            copy.indiv = i.indiv
            copy.distance = i.distance
            copy.cost = i.cost
            next.append(copy)

        next = ga.select(next, tsp)
        for i in range(size):
            j = np.random.randint(0, size)
            next[j].indiv = ga.cross(next[i].indiv, next[j].indiv)
            next[j].indiv = ga.mutate(pop[j].indiv)
        for i in range(size):
            next[i].calculate_objectives(next[i], tsp)

        return next

    def next_pop_improve(self, P, tsp):
        next = []
        indiv = self.indiv
        pop = P
        size = len(pop)
        for i in pop:
            i = indiv.reset_indiv(i)
            copy = indiv.create_indiv(tsp)
            copy.indiv = i.indiv
            copy.distance = i.distance
            copy.cost = i.cost
            next.append(copy)

        next_cross = ga.select(next, tsp) # 交叉前的种群
        next_mutation = Copy.deepcopy(next_cross) # 变异前的种群

        for i in range(size):
            # 交叉及变异
            j = np.random.randint(0, size)
            next[j].indiv = ga.cross(next[i].indiv, next[j].indiv)
            next[j].indiv = ga.mutate(pop[j].indiv)
        for i in range(size):
            # 交叉
            j = np.random.randint(0, size)
            next_cross[j].indiv = ga.cross(next[i].indiv, next[j].indiv)

        # 改进：分别对交叉和变异后的种群进行保存，并与原始种群进行合并
        next.extend(next_cross)
        next.extend(next_mutation)
        for i in range(len(next)):
            next[i].calculate_objectives(next[i], tsp)
        return next

    def next_pop_newmutation(self, P, tsp):
        next = []
        indiv = self.indiv
        pop = P
        size = len(pop)
        for i in pop:
            i = indiv.reset_indiv(i)
            copy = indiv.create_indiv(tsp)
            copy.indiv = i.indiv
            copy.distance = i.distance
            copy.cost = i.cost
            next.append(copy)

        next_cross = ga.select(next, tsp)  # 交叉前的种群
        next_mutation = Copy.deepcopy(next_cross)  # 变异前的种群

        for i in range(size):
            # 交叉及变异
            j = np.random.randint(0, size)
            next[j].indiv = ga.cross(next[i].indiv, next[j].indiv)
            next[j].indiv = ga.mutate_improve(pop[j].indiv)
        for i in range(size):
            # 交叉
            j = np.random.randint(0, size)
            next_cross[j].indiv = ga.cross(next[i].indiv, next[j].indiv)

        # 改进：分别对交叉和变异后的种群进行保存，并与原始种群进行合并
        next.extend(next_cross)
        next.extend(next_mutation)
        for i in range(len(next)):
            next[i].calculate_objectives(next[i], tsp)
        return next

