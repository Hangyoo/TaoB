import random
from NSGA2 import config


#[[个体1]，[个体2]，[]]
def initializePopulation():
    population = []
    for i in range(config.popSize):
        individual = []
        for j in range(44):
            individual.append(random.randint(0,11))
        population.append(individual)
    return population

if __name__ == '__main__':
    population = initializePopulation()
    print(population[0])