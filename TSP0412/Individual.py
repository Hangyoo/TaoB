import numpy as np
from TSP import *


class Individual:
    # The number that p is dominated
    Np = 0
    # dominate list
    Sp = []
    # rank
    rank = 0
    # crowding distance
    crowding_distance = 0
    # city distance in total
    distance = 0
    # city cost in total
    cost = 0

    indiv = np.array([])

    def __init__(self):
        pass

    def create_indiv(self, tsp):
        # create new individual
        i = Individual()
        i.Np = 0
        i.Sp = []
        i.rank = 0
        i.dp = 0
        # add random cities
        g = np.arange(tsp.cities.shape[0])
        np.array(np.random.shuffle(g))
        i.indiv = g

        self.calculate_objectives(i, tsp)

        return i

    def calculate_objectives(self, i, tsp):
        # calculte distance
        i.distance = TSP.get_distance(tsp, i.indiv)
        # calculte cost
        i.cost = TSP.get_allcost(tsp, i.indiv)

    def reset_indiv(self, indiv):
        # reset individual
        indiv.Np = 0
        indiv.Sp = []
        indiv.rank = 0

        return indiv
