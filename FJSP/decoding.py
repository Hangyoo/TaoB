from FJSP import encoding, readtext
from FJSP import gantt

#分割基于机器分配的编码，划分为每个工件所需的机器
def split_ms(parameters, ms):
    jobs_machine = []   #储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ms[current:current+len(job)])
        current += len(job)
    return jobs_machine

#寻找最早可开始加工时间，返回可以最早开始加工的时间
def find_first_available_place(start_ctr, duration, machine_jobs):
    #判断机器空闲是否可用
    def is_free(machine_used, start, duration):
        # machine_used 机器使用列表； start开始时间； duration加工时长
        for k in range(start, start + duration):
            if not machine_used[k]:  # 都为True可行，否则返回False
                return False
        return True

    max_duration_list = []
    max_duration = start_ctr + duration

    # max_duration = start_ctr + duration 或 max(possible starts) + duration
    # 最长时间 = （前工序完工时间+time） 或 （机器可用时间+time）
    if machine_jobs:
        for job in machine_jobs:   #job --('0-1', 6, 0, 0)  （‘’，加工时间，开始时间，）
            max_duration_list.append(job[3] + job[1])

        max_duration = max(max(max_duration_list), start_ctr) + duration

    machine_used = [True] * max_duration  # machine_used  机器可用列表

    # 更新机器可用列表
    for job in machine_jobs:
        start = job[3]
        time = job[1]
        for k in range(start, start + time):
            machine_used[k] = False

    # 寻找满足约束的第一个可用位置
    for k in range(start_ctr, len(machine_used)):
        if is_free(machine_used, k, duration):
            return k  #返回可以开始加工的位置

#对个体进行解码，分配工件至机器。返回每台机器上加工任务
def decode(parameters, os, ms):
    o = parameters['jobs']
    machine_operations = [[] for i in range(parameters['machinesNb'])]   #[[机器1],[],[]..[机器n]]
    ms_s = split_ms(parameters, ms)    # 每个工件的加工机器[[],[],[]..]
    Job_process = [0] * len(ms_s)      # len(ms_s）为工件数,储存第几工件加工第几工序
    Job_before = [0] * len(ms_s)  # 储存工件前一工序的完工时间

    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for job in os:
        index_machine = ms_s[job][Job_process[job]]    #获取工件job的第Job_process[job]工序加工机器
        machine = o[job][Job_process[job]][index_machine]['machine']-1       #（工件，工序，机器序号）加工机器(索引重1开始)
        prcTime = o[job][Job_process[job]][index_machine]['processingTime']  #加工时间

        start_cstr = Job_before[job]  #前工序的加工时间
        # 能动解码
        start = find_first_available_place(start_cstr, prcTime, machine_operations[machine])
        text = "{}-{}".format(job, Job_process[job])  #工件-工序（索引均为0）

        #（‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
        machine_operations[machine].append((text, prcTime, start_cstr, start, start+prcTime, job))
        # 更新工件加工到第几工序
        Job_process[job] += 1
        Job_before[job] = (start + prcTime)
    return machine_operations   # [[(),(),()],[],[]]


def translate_decoded_to_gantt(machine_operations):
    data = {}

    for idx, machine in enumerate(machine_operations):
        machine_name = "Machine-{}".format(idx + 1)
        operations = []
        for operation in machine:
            operations.append([operation[3], operation[3] + operation[1], operation[0]])

        data[machine_name] = operations
    return data

if __name__ == '__main__':

    patch = r'C:\Users\Hangyu\PycharmProjects\FJSPP\柔性作业车间调度\FJSP\FJSPBenchmark\Brandimarte_Data\Mk01.fjs'

    parameters = readtext.Readtext(patch).readtext()

    # #os_ms = encoding.initializePopulation(parameters)[1]
    # os_ms = [[1,1,1,0,0],[0,2,1,1,1]]
    # (os, ms) = os_ms
    # decoded = decode(parameters, os, ms)
    # for x in decoded:
    #     print(x)
    data = translate_decoded_to_gantt(decoded)
    # print(data)
    # gantt.draw_chart(data)
