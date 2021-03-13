import time
import matplotlib.pyplot as plt
from 柔性作业车间调度.FJSP import gantt,  Objective_learn, adapt
from 柔性作业车间调度.Immune import Genetic, Decoding
from 柔性作业车间调度.Immune import Genepopulation, Readtext,select
from 柔性作业车间调度.FJSP import config
from 柔性作业车间调度.Immune.Objective import maxload, TimeTaken, sumload

# 获取算例

parameters = Readtext.readtext(r"C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk05.fjs")
t0 = time.time()
pool = []
for i in range(10):
    population = Genepopulation.initializePopulation(parameters)  #初始化种群
    min_list = []
    avg_list = []


    gen = 1
    while gen < config.maxGen: #若不满足终止条件则循环
        print('第%d次迭代'%gen)
        population = Genetic.selection(population, parameters)
        population = Genetic.crossover(population, parameters)
        population = Genetic.mutation (population, parameters)
        max_fit, min_fit, avg_fit, adapt1 = adapt.adapting(population, parameters)
        min_list.append(min_fit)
        avg_list.append(avg_fit)
        print('本代最优适应度值:', min_fit)
        gen = gen + 1

    sortedPop = sorted(population, key=lambda individual: adapt.fitness(individual,parameters))

    print('个体排序',sortedPop)
    pool.extend(sortedPop[0:10])

t1 = time.time()
total_time = t1 - t0
print("运行时间：{0:.2f}s".format(total_time))

object = []
for p in pool:
    (os,ms,hs) = p
    decode = Decoding.decode(parameters,os,ms,hs)
    f1 = Objective_learn.TimeTaken(decode)
    f2 = Objective_learn.maxload(decode)
    f3 = Objective_learn.sumload(decode)
    object.append([f1,f2,f3])

result = select.select(object)
print(result)



