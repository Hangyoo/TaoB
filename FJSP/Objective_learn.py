

#（柔性+工人）计算最大完工时间
def TimeTaken(decoded):      #个体（[os],[ms]）
    # 获取每台机器上最大完工时间
    max_per_machine = []
    for item in decoded:
        max_d = 0
        for info in item:  # info = （‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
            end = info[4]  #本工件完工时间
            if end > max_d:
                max_d = end
        max_per_machine.append(max_d)
    makespan = round(max(max_per_machine),2)
    return makespan

# （柔性+工人机器最大负荷
def maxload(decoded):
    Time = {i:0 for i in range(len(decoded))}
    i = 0
    for item in decoded:
        for info in item:
            Time[i] += info[1]
        i += 1
    maxload = round(max(Time.values()),2)
    return maxload  # 最大机器负荷

# （柔性+工人）机器总负荷函数
def sumload(decoded):
    Time = {i: 0 for i in range(len(decoded))}
    i = 0
    for item in decoded:
        for info in item:
            Time[i] += info[1]
        i += 1
    sumload = round(sum(Time.values()),2)
    return sumload
