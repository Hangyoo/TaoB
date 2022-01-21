from FJSP_batch import config
from FJSP_batch import genetic

def adapting(population,parameters):
    adapt = []
    for individual in population:
        fit = genetic.timeTaken(individual, parameters)
        adapt.append(fit)
    max_fit = max(adapt)
    min_fit = min(adapt)
    avg_fit = sum(adapt) / config.popSize
    return max_fit,min_fit,avg_fit,adapt