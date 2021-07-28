from FJSP import readtext
from collections import Counter
from FJSP.decoding import split_ms
from FJSP import decoding


# 获取柔性作业车间每个工件的完工时间
def eachmakesapn(os_ms,parameters):
    (os, ms) = os_ms
    n = len(parameters['jobs'])
    Cjob = {job: 0 for job in range(n)}  # 每个工件完工时间
    process = [(len(parameters['jobs'][i]) - 1) for i in range(n)]
    lastprocess = ["{}-{}".format(job, process[job]) for job in range(n)]  # 工件索引为0,工序索引都为0
    decoded = decoding.decode(parameters, os, ms)
    # 获取每台机器上最大完工时间

    for machine in decoded:
        max_d = 0
        for job in machine:  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
            if job[0] in lastprocess:
                Cjob[job[5]] = max_d  # 每个工件最后完工时间
    return Cjob



'''---------------------柔性车间常用目标函数-------------------------------'''
#（柔性）计算最大完工时间
def TimeTaken(os_ms, parameters):      #个体（[os],[ms]）
    (os, ms) = os_ms
    decoded = decoding.decode(parameters, os, ms)

    # 获取每台机器上最大完工时间
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)
    makespan = max(max_per_machine)
    return makespan

#（柔性）计算最大完工时间 + 拖期提前期 （可以带惩罚函数）
def timeTaken(os_ms, parameters, job_priority_duedate):      #个体（[os],[ms]）
    (os, ms) = os_ms
    n = len(parameters['jobs'])
    Cjob = {job: 0 for job in range(n)}  # 每个工件完工时间
    ear_tar = {}  # 记录工件最早最晚到达时间 d_record={job:[earliness time,tardiness time]}
    process = [(len(parameters['jobs'][i])-1) for i in range(n)]
    lastprocess = ["{}-{}".format(job,process[job]) for job in range(n)]  # 工件索引为0,工序索引都为0
    decoded = decoding.decode(parameters, os, ms)
    # 获取每台机器上最大完工时间
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for job in machine:  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
            if job[0] in lastprocess:
                Cjob[job[5]] = max_d   # 每个工件最后完工时间
        max_per_machine.append(max_d)
    makespan = max(max_per_machine)

    # 拖期提前期
    for j in range(n):
        if Cjob[j] > job_priority_duedate[j][1]:  # 完工时间超过交货期
            job_tardiness = Cjob[j] - job_priority_duedate[j][1]  # 拖期
            job_earliness = 0
            ear_tar[j] = [job_earliness, job_tardiness]
        elif Cjob[j] < job_priority_duedate[j][1]:  # 提前期
            job_tardiness = 0
            job_earliness = job_priority_duedate[j][1] - Cjob[j]
            ear_tar[j] = [job_earliness, job_tardiness]
        else:  # 如期
            job_tardiness = 0
            job_earliness = 0
            ear_tar[j] = [job_earliness, job_tardiness]
    # sum(早到时间/优先级 + 晚到时间*优先级)
    twet = sum(
        (ear_tar[j][0] / job_priority_duedate[j][0]) + ear_tar[j][1] * job_priority_duedate[j][0] for j in range(n))
    # twet = sum(ear_tar[j][0] * 提前期惩罚数 + ear_tar[j][1] * 拖期惩罚数 for j in range(n))
    return makespan, twet



