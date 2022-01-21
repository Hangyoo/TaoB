
import time
import matplotlib.pyplot as plt
from FJSP_batch import Genetic_ as genetic
from FJSP_batch import decoding, encoding, readtext, adapt
from FJSP_batch import config

# 获取算例
patch = 'data.fjs'
batch = [3,7,5,1,5,1,3,7]  ## 设置批量 工件1第1批 工件2第1批 工件3第1批 工件4第1批 工件1第2批 工件2第2批 工件3第2批 工件4第2批##
A = readtext.Readtext(patch,batch)
parameters = A.readtext()
t0 = time.time()
population = encoding.initializePopulation(parameters)  #初始化种群
min_list = []
avg_list = []


gen = 1
val = 100000
while gen < config.maxGen: #若不满足终止条件则循环
    print('第%d次迭代'%gen)
    population = genetic.selection(population, parameters)
    population = genetic.crossover(population, parameters)
    population = genetic.mutation (population, parameters)
    max_fit, min_fit, avg_fit, adapt1 = adapt.adapting(population, parameters)
    if avg_fit < val:
        val = avg_fit
    min_list.append(min_fit)
    avg_list.append(val)
    print('本代种群最小完工时间:', min_fit)
    gen = gen + 1

sortedPop = sorted(population, key=lambda individual: genetic.timeTaken(individual, parameters))
makespan = genetic.timeTaken(population[0], parameters)
print('完工时间',makespan)
print('最优染色体：',population[0])

#--------------输出计算时间----------------#
t1 = time.time()
total_time = t1 - t0
print("运行时间：{0:.2f}s".format(total_time))


#---------------绘制迭代图-----------------#
iter = [i for i in range(0, config.maxGen - 1)]
# print(min_list)
# print(avg_list)
plt.plot(iter,avg_list,'--')
plt.plot(iter,min_list,':')
plt.ylabel('Makespan')
plt.xlabel('Iteration')
plt.legend(['best','average'])
plt.legend(['average'])
plt.show()