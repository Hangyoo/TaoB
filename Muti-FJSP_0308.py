import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd


# 读取数据
def read_fjs(filename):
    inf = 10000
    with open(filename) as f:
        lines = f.readlines()
    first_line = lines[0].split()
    n_jobs = int(first_line[0])  # N
    n_machines = int(first_line[1])  # m
    nb_operations = [int(lines[j + 1].split()[0]) for j in range(n_jobs)]  # ni
    n_operations = np.sum(nb_operations)
    nb_tasks = sum(nb_operations[j] for j in range(n_jobs))
    processing_time = [[inf for m in range(n_machines)] for t in range(nb_tasks)]
    # For each job, for each operation, the corresponding task id
    operation_task = [[0 for o in range(nb_operations[j])] for j in range(n_jobs)]
    sum_time = 0
    ni = [0 for _ in range(n_jobs)]
    Ptime = {}
    id = 0
    for j in range(n_jobs):
        line = lines[j + 1].split()
        tmp = 0
        for o in range(nb_operations[j]):
            n_machines_operation = int(line[tmp + o + 1])
            ni[j] += 1
            for i in range(n_machines_operation):
                machine = int(line[tmp + o + 2 * i + 2]) - 1
                time = int(line[tmp + o + 2 * i + 3])

                Ptime[int(j + 1), int(o + 1), int(machine + 1)] = time

                processing_time[id][machine] = time
                sum_time += time
            operation_task[j][o] = id
            id += 1
            tmp += 2 * n_machines_operation

    tasks = []
    for job in range(n_jobs):
        jtasks = operation_task[job]
        for task in jtasks:
            times = processing_time[task]
            for machine in range(n_machines):
                time = times[machine]
                if time < inf:
                    # tasks.append((job, task, machine, time))  # 索引均为从0开始的
                    tasks.append((job + 1, task + 1, machine + 1, time))  # 索引均为从1开始的

    # 工序列表
    ops = np.zeros((n_jobs, 1 + max([len(jtasks) for jtasks in operation_task])))
    for job in range(n_jobs):
        jtasks = operation_task[job]
        for j in range(len(jtasks)):
            ops[job][0] = job + 1  # 工件
            ops[job][j + 1] = jtasks[j] + 1  # 工件包含的工序

    columns = ['Operation', 'Machine']
    for job in range(n_jobs):
        columns.append('J%d' % (job + 1))
    data = np.zeros((len(tasks), len(columns)))
    Processing_time = pd.DataFrame(data, columns=columns, dtype=np.int32)

    for i in range(len(tasks)):
        job, task, machine, time = int(tasks[i][0]), int(tasks[i][1]), int(tasks[i][2]), int(tasks[i][3])

        Processing_time.iloc[i, 0] = task
        Processing_time.iloc[i, 1] = machine

        Processing_time.iloc[i, job + 1] = time
    return np.array(tasks), n_jobs, n_machines, n_operations, sum_time, ops, Processing_time, Ptime, ni


# 用于寻找可以处理job的第opr个工序可叫加工的机器
def findMac(job, opr, Ptime):
    macList = []
    for item in Ptime.keys():
        i, j, k = item
        if job == i and opr == j:
            macList.append(k)  # 对变量-1均想将索引变为从0开始
    return set(macList)


filename = r'D:\文档\xianyu_code\22.03.08\Mk01.fjs'
tasks, n_jobs, n_machines, n_operations, sum_time, ops, Processing_time, Ptime, ni = read_fjs(filename)

job_num = len(ops)  # 10个工件 N  #i
N = [i + 1 for i in range(job_num)]
Ni = [0, 6, 5, 5, 5, 6, 6, 5, 5, 6, 6]  # 每个工件所包含的工序数目 ni #j

M = [i for i in range(1, n_machines + 1)]  # 10台机器 m #k
n_workers = n_machines  # 工人数 = 机器数 = 10 w
W = [i for i in range(1, n_workers + 1)]  # l = 1 to w #l

M = 1e10  # Big M

m = gp.Model("FJSP")  # create model

# create variables Xijkl
X = []
for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(1, Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        for r in range(i, job_num + 1):  # 对设备k进行循环  k=1,2，..m
            for s in range(1, Ni[r] + 1):
                X.append((i, j, r, s))
X = m.addVars(X, vtype=GRB.BINARY, name="X")

# create variables Yijkl
Y = []
for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(1, Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        for k in range(1, n_machines + 1):  # 对设备k进行循环  k=1,2，..m
            for l in range(1, n_workers + 1):
                Y.append((i, j, k, l))
Y = m.addVars(Y, vtype=GRB.BINARY, name="Y")
# create variables Cij
C = []
for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        C.append((i, j))
C = m.addVars(C, lb=0, vtype=GRB.CONTINUOUS, name="C")  # 每个工件的完工时间

Cmax = m.addVar(vtype=GRB.CONTINUOUS, name="Cmax")  # 最大完工时间（目标函数）

# Set objective
m.setObjective(Cmax, GRB.MINIMIZE)

for i in range(1, job_num + 1):
    C[i, 0] = 0

# add constraints 1
for i in range(1, job_num + 1):
    for j in range(1, Ni[i] + 1):
        Mij = findMac(i, j, Ptime)
        m.addConstr(sum(sum(Y[i, j, k, l] for k in Mij) for l in W) == 1)

# Add constraints 2
for i in range(1, job_num):
    for j in range(1, Ni[i]):
        Mij = findMac(i, j, Ptime)
        m.addConstr(C[i, j] >= C[i, j - 1] + sum(sum(Y[i, j, k, l] * Ptime[i, j, k] for k in Mij) for l in W))

# Add constraints 3
for i in range(1, job_num):
    for j in range(1, Ni[i] + 1):
        Mij = findMac(i, j, Ptime)
        for r in range(1, job_num + 1):
            for s in range(1, Ni[r] + 1):
                Mrs = findMac(r, s, Ptime)
                if i < job_num and i < j and r > i and r > s:
                    for k in list(set(Mij).intersection(set(Mrs))):
                        m.addConstr(C[i, j] >= C[r, s] + sum(Y[i, j, k, l] * Ptime[i, j, k] for l in W) - M * (
                                1 - X[i, j, r, s]) - M * (
                                            2 - sum(Y[i, j, k, l] for l in W) - sum(Y[r, s, k, l] for l in W)))

# Add constraints 4
for i in range(1, job_num):
    for j in range(1, Ni[i] + 1):
        Mij = findMac(i, j, Ptime)
        for r in range(1, job_num + 1):
            for s in range(1, Ni[r] + 1):
                Mrs = findMac(r, s, Ptime)
                if i < job_num and i < j and r > i and r > s:
                    for k in list(set(Mij).intersection(set(Mrs))):
                        m.addConstr(
                            C[r, s] >= C[i, j] + sum(Y[r, s, k, l] * Ptime[r, s, k] for l in W) - M * X[
                                i, j, r, s] - M * (
                                    2 - sum(Y[i, j, k, l] for l in W) - sum(Y[r, s, k, l] for l in W)))
# Add constraints 5
for i in range(1, job_num):
    for j in range(1, Ni[i] + 1):
        Mij = findMac(i, j, Ptime)
        for r in range(1, job_num + 1):
            for s in range(1, Ni[r] + 1):
                Mrs = findMac(r, s, Ptime)
                if i < job_num and i < j and r > i and r > s:
                    for l in W:
                        m.addConstr(
                            C[i, j] >= C[r, s] + sum(Y[i, j, k, l] * Ptime[i, j, k] for k in Mij) - M * (
                                    1 - X[i, j, r, s]) - M * (
                                    2 - sum(Y[i, j, k, l] for k in Mij) - sum(Y[r, s, k, l] for k in Mrs)))

# Add constraints 6
for i in range(1, job_num):
    for j in range(1, Ni[i] + 1):
        Mij = findMac(i, j, Ptime)
        for r in range(1, job_num + 1):
            for s in range(1, Ni[r] + 1):
                Mrs = findMac(r, s, Ptime)
                if i < job_num and i < j and r > i and r > s:
                    for l in W:
                        m.addConstr(
                            C[r, s] >= C[i, j] + sum(Y[r, s, k, l] * Ptime[r, s, k] for k in Mrs) - M * X[
                                i, j, r, s] - M * (
                                    2 - sum(Y[i, j, k, l] for k in Mij) - sum(Y[r, s, k, l] for k in Mrs)))

# Add constraints 7
for i in range(1, job_num + 1):
    m.addConstr(Cmax >= C[i, Ni[i]])

# Optimize model
m.optimize()
m.write('Multi_FJSP.lp')

# 输出有意义的结果
for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(1, Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        for r in range(i, job_num + 1):  # 对设备k进行循环  k=1,2，..m
            for s in range(1, Ni[r] + 1):
                if X[i, j, r, s].x == 1:
                    print('X[%d,%d,%d,%d]' % (i, j, r, s), X[i, j, r, s].x)

for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(1, Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        for k in range(1, n_machines + 1):  # 对设备k进行循环  k=1,2，..m
            for l in range(1, n_workers + 1):
                if Y[i, j, k, l].x == 1:
                    print('Y[%d,%d,%d,%d]' % (i, j, k, l), Y[i, j, k, l].x)

for i in range(1, job_num + 1):  # 对工件Ji进行循环
    for j in range(1,Ni[i] + 1):  # 对工件Ji 所含有的ni个工序进行循环
        if C[i, j].x > 0:
            print('C[%d,%d]' % (i, j), C[i, j].x)

print('Obj: \t  %g' % m.objVal)

# 输出所有结果
"""
for v in m.getVars():
    print('%s \t %g' % (v.varName, v.x))

print('Obj: \t  %g' % m.objVal)
"""
