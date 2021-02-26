import random
from taobao import config, readtext

#产生基于工序的编码(OS)
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

# 基于机器分配的编码(MS)
def generateMS(parameters):
    jobs = parameters['jobs']
    MS = []
    for job in jobs:
        for op in job:
            randomMachine = random.randint(0, len(op)-1)   #工件索引从0开始，randint包含上界
            MS.append(randomMachine)
    return MS

# 人员编码
def generateWS(parameters,MS):
    #set = {1: [2, 3, 4], 2: [2, 3, 4], 3: [2, 3, 4], 4: [1, 2], 5: [1, 2], 6: [1, 5], 7: [1, 5], 8: [5, 6], 9:[0], 10:[0], 11:[0]}

    ms = []
    current = 0
    for index, job in enumerate(parameters['jobs']):
        temp = MS[current:current + len(job)]  # len(job) 工序数
        current += len(job)
        for i in range(len(job)):
            # 每个工序选择的机器
            ms.append(job[i][temp[i]]['machine'])
    WS = []

    # 为机器选择合适的员工
    for o in ms:
        k = random.choice([i for i in range(parameters['machinesNb'])])
        WS.append(k)
    return WS

#[[个体1]，[个体2]，[]]
def initializePopulation(parameters):
    population = []
    for i in range(config.popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        HS = generateWS(parameters,MS)
        population.append([OS, MS, HS])
    return population

if __name__ == '__main__':
    patch = r'C:\Users\Hangyu\PycharmProjects\FJSPP\taobao\Benchmark\Mk01.fjs'
    parameters = readtext.Readtext(patch).readtext()
    population = initializePopulation(parameters)
    print(population[0])