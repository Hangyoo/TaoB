#分割基于机器分配的编码，划分为每个工件所需的机器
def split_ms(parameters, ms):
    jobs_machine = []   #储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ms[current:current+len(job)])
        current += len(job)
    return jobs_machine

# 划分为每个工件所需的人员
def split_ws(parameters, ws):
    jobs_machine = []   #储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ws[current:current+len(job)])
        current += len(job)
    return jobs_machine


#对个体进行解码，分配工件至机器。返回每台机器上加工任务
def decodeBenchmark(parameters, os, ms, ws):
    o = parameters['jobs']
    machine_operations = [[] for i in range(parameters['machinesNb'])]   #[[机器1],[],[]..[机器n]]
    Job_process = [0] * parameters['jobsnum']      # 储存第几工件加工第几工序
    Job_before = [0] * parameters['jobsnum']       # 储存工件前一工序的完工时间
    Machine_before = [0]*parameters['machinesNb']  # 储存工件前一工序的完工时间
    People_before = [0] * parameters['machinesNb'] # 储存工人前一工序的完工时间

    # 储存上一个工件，和上一个工序
    before_i = [0]*parameters['machinesNb']
    before_j = 0

    ni = []  # 存储每个工件的工序数
    for job in parameters['jobs']:
        ni.append(len(job))

    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for i in range(len(os)):
        job = os[i]
        opr = Job_process[job]
        index_machine = ms[sum(ni[:job])+opr]  # 获取Oij的加工机器
        # todo
        machine = o[job][opr][index_machine]['machine']-1       #（工件，工序，机器序号）加工机器(索引重1开始)
        people = ws[sum(ni[:job])+opr]
        prcTime = o[job][opr][index_machine]['processingTime']  #加工时间

        start_cstr = Job_before[job]  #前工序的加工时间
        start_machine = Machine_before[machine]
        before_finishtime = People_before[people]
        start = max(start_cstr, start_machine, before_finishtime)

        text = "{}-{}-{}".format(job, opr, people)  #工件-工序（索引均为0）

        #（‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,机器号，人员号）
        machine_operations[machine].append((text, prcTime, start_cstr, start, start+prcTime, machine, people))
        # 更新工序
        Job_process[job] += 1
        # 加工时间
        Job_before[job] = start + prcTime
        Machine_before[machine] = start + prcTime
        People_before[people] = start + prcTime
        # 历史加工任务
        before_i[machine] = job

    return machine_operations   # [[(),(),()],[],[]]


# 绘制甘特图时使用
def translate_decoded_to_gantt(machine_operations):
    data = {}

    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        for operation in machine:
            starttime = operation[3]
            endtime = operation[3] + operation[1]
            label = operation[0]
            operations.append([starttime, endtime, label])

        data[machine_name] = operations
    return data

