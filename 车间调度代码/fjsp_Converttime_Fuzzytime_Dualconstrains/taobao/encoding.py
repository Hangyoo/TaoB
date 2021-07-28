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
def generateWS(parameters,OS,MS):
    # 方法1：若指定某台机器由哪几个工人进行加工，可在此处设定
    # machine_list = decode_machine(parameters, OS, MS)
    # WS = []
    # set = {0: [0, 2, 3], 1: [2, 3, 1], 2: [2, 3, 0], 3: [1, 2]}  # 机器索引从0开始，工人索引从0开始，且工人数不超过机器数
    # for o in machine_list:
    #     count = random.choice(set[o])
    #     WS.append(count)

    #--------------------------------------------------------
    # 方法2：默认员工可操作所有机器
    WS = []
    #为机器选择合适的员工
    for o in MS:
        workerlist = [i for i in range(parameters['machinesNb'])]
        k = random.choice(workerlist)
        WS.append(k)
    return WS


#[[个体1]，[个体2]，[]]
def initializePopulation(parameters):
    population = []
    for i in range(config.popSize):
        OS = generateOS(parameters)
        MS = generateMS(parameters)
        WS = generateWS(parameters,OS,MS)
        population.append([OS, MS, WS])
    return population

#对个体进行解码，分配工件至机器。返回每台机器上加工任务
def decode_machine(parameters, os, ms):
    o = parameters['jobs']
    machine_list = []
    Job_process = [0] * parameters['jobsnum']      # 储存第几工件加工第几工序

    ni = []  # 存储每个工件的工序数
    for job in parameters['jobs']:
        ni.append(len(job))

    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for i in range(len(os)):
        job = os[i]
        opr = Job_process[job]
        index_machine = ms[sum(ni[:job])+opr]  # 获取Oij的加工机器
        machine = o[job][opr][index_machine]['machine']-1       #（工件，工序，机器序号）加工机器(索引重1开始)
        # 更新工序
        Job_process[job] += 1
        machine_list.append(machine)

    return machine_list   # [[(),(),()],[],[]]

