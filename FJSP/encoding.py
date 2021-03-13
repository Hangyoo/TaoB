import random
from FJSP import config
from FJSP.readtext import Readtext


#产生基于工序的编码
def generateOS(parameters):
    jobs = parameters['jobs']
    OS = []
    i = 0               #工件索引从0开始
    for job in jobs:    #获取工件数
        for op in job:  #获取工件的工序数
            OS.append(i)
        i = i+1
    random.shuffle(OS)

    return OS

# 基于机器分配的编码
def generateMS(parameters):
    jobs = parameters['jobs']
    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op)-1)   #工件索引从0开始，randint包含上界
            MS.append(randomMachine)
    return MS

#工序编码+机器编码
#[[个体1]，[个体2]，[]]
def initializePopulation(parameters):
    gen = []
    for i in range(100):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        gen.append([OS, MS])
    return gen

if __name__ == '__main__':
    patch = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk09.fjs'

    parameters = Readtext(patch).readtext()
    population = initializePopulation(parameters)
    print(population)