from FJSP_batch import decoding


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



