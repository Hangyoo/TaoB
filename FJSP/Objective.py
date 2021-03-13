from 柔性作业车间调度.FJSP import readtext
from collections import Counter
from 柔性作业车间调度.FJSP.decoding import split_ms
from 柔性作业车间调度.FJSP import decoding


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

# （作业车间）返回最大完工时间 和 提前期与拖期的值（可添加惩罚函数）
def makespan_tardiness(n,m,individual,T,Jm,job_priority_duedate):
    count = [0] * n  # 各工件工序
    Cjob = {key: 0 for key in range(n)}  # 工件完工时间
    Cmac = {key: 0 for key in range(0, m)}  # 机器完工时间
    TPval = [[0]*n*m,[0]*n*m]   # 记录每个工件的开工时间和完工时间
    ear_tar = {}  # 记录工件最早最晚到达时间 d_record={job:[earliness time,tardiness time]}
    k = 0
    for j in individual:  # 获取个体
        time = int(T[j][count[j]])  # pt[i] 工件 --- [key_count[i]]  工序  的加工时间
        machine = int(Jm[j][count[j]])
        TPval[0][k] = max(Cjob[j],Cmac[machine])
        Cjob[j] = Cjob[j] + time  # 工件完工时间
        Cmac[machine] = Cmac[machine] + time  # 机器完工时间
        if Cmac[machine] < Cjob[j]:  # 取两者最大值
            Cmac[machine] = Cjob[j]
            TPval[1][k] = Cjob[j]
        else:
            Cjob[j] = Cmac[machine]
            TPval[1][k] = Cmac[machine]
        k += 1
        count[j] += 1
    makespan = max(Cjob.values())

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
    twet = sum((ear_tar[j][0] / job_priority_duedate[j][0]) + ear_tar[j][1] * job_priority_duedate[j][0] for j in range(n))
    #twet = sum(ear_tar[j][0] * 提前期惩罚数 + ear_tar[j][1] * 拖期惩罚数 for j in range(n))
    return makespan, twet

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

# 机器最大负荷（柔性） 非柔性的都一边大
def maxload(os_ms,parameters):
    mac = []
    os,ms = os_ms[0],os_ms[1]
    jobs_machine = []  # 储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ms[current:current + len(job)])
        current += len(job)

    Job_process = [0] * len(jobs_machine)  # len(ms_s）为工件数,储存第几工件加工第几工序
    o = parameters['jobs']
    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for job in os:
        index_machine = jobs_machine[job][Job_process[job]]  # 获取工件job的第Job_process[job]工序加工机器
        machine = o[job][Job_process[job]][index_machine]['machine'] - 1
        mac.append(machine)
        Job_process[job] += 1
    long = len(list(set(mac)))
    temp = Counter(mac).most_common(long)  # [(5, 18), (2, 16), (6, 13), (3, 12), (0, 11), (4, 7), (1, 7), (7, 6)]  机器编号(0开始)+次数
    return temp[0][1]  # 最大机器负荷

# 机器总负荷函数
def sumload(os_ms, parameters):
    mac = []
    os, ms = os_ms[0], os_ms[1]
    jobs_machine = []  # 储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ms[current:current + len(job)])
        current += len(job)

    Job_process = [0] * len(jobs_machine)  # len(ms_s）为工件数,储存第几工件加工第几工序
    o = parameters['jobs']
    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for job in os:
        index_machine = jobs_machine[job][Job_process[job]]  # 获取工件job的第Job_process[job]工序加工机器
        machine = o[job][Job_process[job]][index_machine]['machine'] - 1
        mac.append(machine)
        Job_process[job] += 1
    long = len(list(set(mac)))
    temp = Counter(mac).most_common(long)  # [(5, 18), (2, 16), (6, 13), (3, 12), (0, 11), (4, 7), (1, 7), (7, 6)]
    sumload = sum(temp[i][1] for i in range(long))
    return sumload

# 平均流程时间
def avgflowtime(os_ms,parameters,R): # P 为每个工件的释放时间 [1,2,6,3,20,...]
    n = len(parameters['jobs'])
    Cjob = eachmakesapn(os_ms,parameters)
    gap = [(Cjob[i]-R[i]) for i in range(n)]
    flowtime = sum(gap)/n
    return round(flowtime,3)  # 保留3位小数

# 总拖期 (提前完工不算，滞后才算)
def sumtar(os_ms, parameters, job_priority_duedate):
    n = len(parameters['jobs'])
    Cjob = eachmakesapn(os_ms, parameters)
    tardiness = []
    for i in range(n):
        tar = Cjob[i] - job_priority_duedate[i][1]
        if tar > 0:
            tardiness.append(tar)
        else:
            pass
    sumtar = sum(tardiness)
    return sumtar

# 费用（静态费率和动态费率）
def cost(os_ms, parameters, MC, MD):
    # MC 原材料费用 [1,2,3,5,9,..] （等于工件个数）
    # MD 机器单位费率  [12,51,48，..]   （等于机器个数）
    (os,ms) = os_ms
    o = parameters['jobs']
    # Time = {0: 95, 1: 9, 2: 45, 3: 52, 4: 26, 5: 56, 6: 67, 7: 45}  汇总每个机器加工时长
    Time = {i:0 for i in range(parameters['machinesNb'])}
    ms_s = split_ms(parameters, ms)  # 每个工件的加工机器[[],[],[]..]
    Job_process = [0] * len(ms_s)  # len(ms_s）为工件数,储存第几工件加工第几工序
    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for job in os:
        index_machine = ms_s[job][Job_process[job]]  # 获取工件job的第Job_process[job]工序加工机器
        machine = o[job][Job_process[job]][index_machine]['machine'] - 1  # （工件，工序，机器序号）加工机器(索引重1开始)
        prcTime = o[job][Job_process[job]][index_machine]['processingTime']  # 加工时间
        Time[machine] += prcTime
        Job_process[job] += 1
    print(Time)
    sumcost = sum(MC) + sum([Time[key]*MD[key] for key in Time.keys()])
    return sumcost


if __name__ == '__main__':
    parameters = readtext.readtext(r"C:\Users\LouHangYu\PycharmProjects\联系\FJSP\FJSPBenchmark\Brandimarte_Data\Mk04.fjs")
    individual = [[12, 2, 8, 10, 7, 5, 5, 12, 11, 12, 1, 1, 7, 4, 4, 13, 8, 5, 11, 13, 0, 6, 5, 4, 10, 14, 0, 9, 8, 0, 5, 6, 9, 1, 9, 3, 3, 0, 10, 0, 14, 14, 13, 1, 2, 9, 5, 8, 4, 11, 5, 1, 9, 4, 3, 7, 2, 6, 14, 3, 2, 6, 4, 0, 0, 0, 8, 8, 3, 8, 12, 11, 7, 5, 6, 1, 11, 14, 11, 4, 2, 7, 5, 8, 7, 1, 8, 10, 2, 14],
                  [0, 1, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 2, 1, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 2, 0, 1, 0]]
    duedate = [[1, 49], [1, 84], [1, 97], [1, 79], [1, 81], [1, 82], [1, 74], [1, 89], [1, 97], [1, 48], [1, 97],
               [1, 84], [1, 64], [1, 39], [1, 95]]
    R = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 工件释放时间
    mac = maxload(individual,parameters)
    sumload = sumload(individual,parameters)
    makespan,tewt = timeTaken(individual,parameters,duedate)
    flowtime = avgflowtime(individual, parameters, R)
    sumtar = sumtar(individual, parameters, duedate)
    MC = [100]*15  # 原材料成本
    MD = [15,20,24,11,15,20,25,21]    # 机器单位时间费用
    sumcost = cost(individual, parameters,MC,MD)

    print('最大(瓶颈)机器负荷：',mac)
    print('总机器负荷：',sumload)
    print('平均流程时间：',flowtime)
    print('总拖期：',sumtar)
    print('最大完工时间：',makespan)
    print('总费用为：', sumcost)

    # 绘制甘特图
    #decoded = decoding.decode(parameters, individual[0], individual[1])
    #data = decoding.translate_decoded_to_gantt(decoded)
    #gantt.draw_chart(data)
