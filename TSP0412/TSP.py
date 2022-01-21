import numpy as np
import pandas as pd
import random
from TSP0412.Individual import *


class TSP:
    # city name
    city_name = np.array([])
    # city location
    cities = np.array([])
    # all cost array
    costs = []
    # distance
    distance = 0
    # cost
    cost = 0

    def __init__(self):
        pass

    def init(self):
        tsp = self
        tsp.load_cities()
        tsp.caculate_costs()

    def load_cities(self, file='berlin52.csv'):
        # load cities
        data = pd.read_csv(file).values
        self.city_name = data[:, :1]
        self.cities = data[:, 1:]

    def caculate_distance(self, city1, city2):
        # calculate distance from city1 to city2
        d = np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

        return d

    def caculate_cost(self, city1, city2):
        # calculate cost from city1 to city2
        c = self.caculate_distance(city1, city2) * (0.6 + random.random())
        #temp1 = self.caculate_distance(city1, city2)
        return c

    def caculate_costs(self):
        # calculate all cost
        for i in range(len(self.cities)):
            c = []
            for j in range(len(self.cities)):
                d = self.caculate_cost(self.cities[i], self.cities[j])
                c.append(d)

            self.costs.append(c)

    def get_cost(self, name1, name2):
        # get cost from city1 to city2
        c = self.costs[name1][name2]

        return c

    def get_distance(self, indiv):
        # calculate individual distance
        total = 0
        for i in range(len(indiv) - 1):
            d = self.caculate_distance(self.cities[indiv[i]], self.cities[indiv[i + 1]])
            total = total + d
        # return to origin
        total = total + self.caculate_distance(self.cities[indiv[-1]], self.cities[indiv[0]])
        return total

    def get_allcost(self, indiv):
        # calculate individual cost
        total = 0
        for i in range(len(indiv) - 1):
            c = self.get_cost(indiv[i], indiv[i + 1])
            total = total + c
        # return to origin
        total = total + self.get_cost(indiv[-1], indiv[0])
        return total


