# 该脚本包含了所提出的混合算法的高级概述
# 严格的描述了文章4.1部分的内容
import time
import matplotlib.pyplot as plt
from FJSP import gantt
from FJSP import Genetic_ as genetic
from FJSP import decoding, encoding, readtext, adapt
from FJSP import config

# 获取算例
patch = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk01.fjs'
A = readtext.Readtext(patch)
parameters = A.readtext()
t0 = time.time()
population = encoding.initializePopulation(parameters)  #初始化种群
min_list = []
avg_list = []


gen = 1
while gen < config.maxGen: #若不满足终止条件则循环
    print('第%d次迭代'%gen)
    population = genetic.selection(population, parameters)
    population = genetic.crossover(population, parameters)
    population = genetic.mutation (population, parameters)
    max_fit, min_fit, avg_fit, adapt1 = adapt.adapting(population, parameters)
    min_list.append(min_fit)
    avg_list.append(avg_fit)
    print('本代最优完工时间:', min_fit)
    gen = gen + 1

sortedPop = sorted(population, key=lambda individual: genetic.timeTaken(individual, parameters))
makespan = genetic.timeTaken(population[0], parameters)
print('完工时间',makespan)

#--------------输出计算时间----------------#
t1 = time.time()
total_time = t1 - t0
print("运行时间：{0:.2f}s".format(total_time))

#---------------绘制甘特图-----------------#
gantt_data = decoding.translate_decoded_to_gantt(decoding.decode(parameters, sortedPop[0][0], sortedPop[0][1]))
gantt.draw_chart(gantt_data)    #输出甘特图
print(sortedPop[0])

#---------------绘制迭代图-----------------#
iter = [i for i in range(0, config.maxGen - 1)]
print(min_list)
print(avg_list)
plt.plot(iter,min_list,'--')
plt.plot(iter,avg_list,':')
plt.ylabel('Makespan')
plt.xlabel('Iteration')
plt.legend(['best','average'])
plt.show()