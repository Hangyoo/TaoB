from taobao import encoding, readtext
from taobao import covertmatrics
import random

#分割基于机器分配的编码，划分为每个工件所需的机器
def split_ms(parameters, ms):
    jobs_machine = []   #储存每个工件每个工序的加工机器(机器列表中位置)[[],[],[]]
    current = 0
    for index, job in enumerate(parameters['jobs']):
        jobs_machine.append(ms[current:current+len(job)])
        current += len(job)
    return jobs_machine


def decodeBenchmark(parameters, os, ms):

    o = parameters['jobs']
    machine_operations = [[] for i in range(parameters['machinesNb'])]   #[[机器1],[],[]..[机器n]]
    ms_s = split_ms(parameters, ms)    # 每个工件的加工机器[[],[],[]..]
    Job_process = [0] * len(ms_s)      # len(ms_s）为工件数,储存第几工件加工第几工序
    Job_before = [0] * len(ms_s)        # 储存工件前一工序的完工时间
    Machine_before = [0]*parameters['machinesNb']

    ni = []
    for job in parameters['jobs']:
        ni.append(len(job))

    # 对基于工序的编码进行依次解码，并安排相应的加工机器
    for i in range(len(os)):
        job = os[i]
        index_machine = ms_s[job][Job_process[job]]    #获取工件job的第Job_process[job]工序加工机器
        machine = o[job][Job_process[job]][index_machine]['machine']-1       #（工件，工序，机器序号）加工机器(索引重1开始)
        prcTime = o[job][Job_process[job]][index_machine]['processingTime']  #加工时间

        start_cstr = Job_before[job]  #前工序的加工时间
        start_machine = Machine_before[machine]
        start = max(start_cstr, start_machine)

        text = [job, Job_process[job]]  #工件-工序（索引均为0）

        #（‘工件-工序’，加工时间，前序完工时间，最早开始时间, 完工时间,工件号）
        machine_operations[machine].append((text, prcTime, start_cstr, start, start+prcTime, job))
        # 更新工件加工到第几工序
        Job_process[job] += 1
        Job_before[job] = (start + prcTime)
        Machine_before[machine] = start + prcTime

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

