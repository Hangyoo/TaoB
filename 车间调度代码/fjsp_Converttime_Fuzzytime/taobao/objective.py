from collections import Counter
from taobao import decoding


'''---------------------柔性车间常用目标函数-------------------------------'''
#（柔性）计算最大完工时间
def TimeTaken(os_ms, parameters, convert):      #个体（[os],[ms]）
    os,ms = os_ms[0],os_ms[1]
    decoded = decoding.decode(parameters, os, ms, convert)

    # 获取每台机器上最大完工时间
    max_per_machine = []
    for machine in decoded:
        # 计算在此零件加工顺序下的切线时间
        max_d = 0
        for i in range(len(machine)):  # job = （‘工件-工序’，加工时间，前序完工时间，最早开始时间）
            job = machine[i]
            end = job[3] + job[1]
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)
    makespan = max(max_per_machine)
    return makespan

#（模糊）计算完工时间
def FuzzyTimeTaken(os_ms, parameters, convert, num):      #个体（[os],[ms]）
    os,ms = os_ms[0],os_ms[1]
    decoded = decoding.Fuzzydecode(parameters, os, ms, convert, num)
    # 获取每台机器上最大完工时间
    max_per_machine = []
    for machine in decoded:
        # 计算在此零件加工顺序下的切线时间
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
def maxload(os_ms,parameters, convert):
    decoded = decoding.decode(parameters, os_ms[0], os_ms[1], convert)
    mac = [0] * parameters['machinesNb']  # 记录每台设备上的工作负荷
    for i in range(parameters['machinesNb']):
        machine_info = decoded[i]
        for item in machine_info:
            mac[i] += item[1]
    maxload = max(mac)
    maxload = round(maxload, 2)
    return maxload  # 最大机器负荷

# 机器总负荷函数
def sumload(os_ms, parameters, convert):
    decoded = decoding.decode(parameters, os_ms[0], os_ms[1], convert)
    mac = [0] * parameters['machinesNb']  # 记录每台设备上的工作负荷
    for i in range(parameters['machinesNb']):
        machine_info = decoded[i]
        for item in machine_info:
            mac[i] += item[1]
    sumload = sum(mac)
    sumload = round(sumload, 2)
    return sumload
