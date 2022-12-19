from collections import Counter
from FJSP_2obj import decoding


'''---------------------柔性车间Benchmark目标函数-------------------------------'''
#（柔性）计算最大完工时间
def TimeTakenBenchmark(os_ms, parameters):      #个体（[os],[ms]）
    os,ms = os_ms[0],os_ms[1]
    decoded = decoding.decodeBenchmark(parameters, os, ms)

    # 获取每台机器上最大完工时间
    max_per_machine = []
    for machine in decoded:
        max_d = 0
        for i in range(len(machine)):  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
            job = machine[i]
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)
    makespan = max(max_per_machine)
    return makespan

# 机器最大负荷（柔性） 非柔性的都一边大
def maxloadBenchmark(os_ms,parameters):
    decoded = decoding.decodeBenchmark(parameters, os_ms[0], os_ms[1])
    mac = [0] * parameters['machinesNb']  # 记录每台设备上的工作负荷
    for i in range(parameters['machinesNb']):
        machine_info = decoded[i]
        for item in machine_info:
            mac[i] += item[1]
    maxload = max(mac)
    return maxload  # 最大机器负荷

# 机器总负荷函数
def sumloadBenchmark(os_ms, parameters):
    decoded = decoding.decodeBenchmark(parameters, os_ms[0], os_ms[1])
    mac = [0] * parameters['machinesNb']  # 记录每台设备上的工作负荷
    for i in range(parameters['machinesNb']):
        machine_info = decoded[i]
        for item in machine_info:
            mac[i] += item[1]
    sumload = sum(mac)
    return sumload

# （柔性+工人）费用（静态费率和动态费率）
def cost(os_ms, parameters, DL, MC, MD):
    # DL 交货期 [300,420,420,480,420,360,360,360] 每个工件的交货时间
    # MC 库存成本 [10,10,10,10,10,..] （等于工件个数）
    # MD 延期成本 [20,20,20，..] （等于机器个数）
    # Time = {0: 95, 1: 9, 2: 45, 3: 52, 4: 26, 5: 56, 6: 67, 7: 45}  汇总每个机器加工时长

    os, ms = os_ms[0], os_ms[1]
    decoded = decoding.decodeBenchmark(parameters, os, ms)
    # 获取每个工件的最大完工时间
    jobs_complete = [0 for _ in range(20)]
    for machine in decoded:
        for i in range(len(machine)):  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
            job = machine[i]
            job_idx, opr = job[0]
            if job[3] + job[1] > jobs_complete[job_idx]:
                jobs_complete[job_idx] = job[3] + job[1]

    cost = 0
    for i in range(len(DL)):
        gap = DL[i] - jobs_complete[i]
        if gap > 0: # 工件先于交货期完成
            cost += gap*MC[i]
        if gap < 0: # 工件拖期
            cost += -gap * MD[i]

    return cost
