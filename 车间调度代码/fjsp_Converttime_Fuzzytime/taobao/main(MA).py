from taobao import covertmatrics
from taobao import encoding, genetic, objective, readtext, decoding
from taobao import config, Non_dominated_sorting, gantt
from taobao.record import record
import matplotlib.pyplot as plt
import time
import pickle

""" ================= 传统FJSP+切线时间 ======================"""

''' ================= 初始参数设定 ======================'''


# 初始参数设置

def main(parameters, convert):
    # 随机生成切线时间并保存到”切线时间.txt“中

    gen = 0
    NIND, MAXGEN, pc, pm = config.popSize, config.maxGen, config.pc, config.pm
    population = encoding.initializePopulation(parameters)
    best_list, best_obj = [], []

    convergence = {"makespan": [], "maxload": [], "sumload": []}
    before_value = [1e10, 1e10, 1e10]

    while gen < MAXGEN:
        print('第%d次迭代' % gen)
        offspring_crossover = genetic.crossover(population, parameters)
        offspring_mutation = genetic.mutation(offspring_crossover, parameters)
        # 对3个种群进行合并, 变成3倍种群的
        Chrom = population + offspring_crossover + offspring_mutation
        chroms_obj_record = {}

        for i in range(NIND * 3):
            os_ms = Chrom[i]
            makespan = objective.TimeTaken(os_ms, parameters, convert)  # 最大完工时间
            maxload = objective.maxload(os_ms, parameters, convert)  # 最大负荷
            sumload = objective.sumload(os_ms, parameters, convert)  # 总负荷
            chroms_obj_record[i] = [makespan, maxload, sumload]

        # 记录每代最小目标值
        front = Non_dominated_sorting.Non_donminated_sorting_MA(NIND, chroms_obj_record)
        population, new_popindex = Non_dominated_sorting.Selection(NIND, front, chroms_obj_record, Chrom)

        chroms_obj_record_new = {}
        for i in new_popindex:
            chroms_obj_record_new[i] = chroms_obj_record[i]
        convergence, before_value = record(chroms_obj_record_new, convergence, before_value)

        if gen == MAXGEN - 1:
            for i in front[0]:
                best_list.append(Chrom[i])
                best_obj.append(chroms_obj_record[i])
        gen += 1

    return best_obj, best_list, convergence


def check(best_obj, best_list):
    list0 = []
    list1 = []
    # 对重复目标值和个体进行筛选
    for i in range(len(best_obj)):
        if best_obj[i] in list0:
            pass
        else:
            list0.append(best_obj[i])
            list1.append(best_list[i])

    return list0, list1


def cal_fuzzy_time(best_list, parameters, convert):
    obj_record = []
    for os_ms in best_list:
        C_min = objective.FuzzyTimeTaken(os_ms, parameters, convert, 0)
        C_avg = objective.FuzzyTimeTaken(os_ms, parameters, convert, 1)
        C_max = objective.FuzzyTimeTaken(os_ms, parameters, convert, 2)
        makespan = (C_min, C_avg, C_max)
        maxload = objective.maxload(os_ms, parameters, convert)  # 最大负荷
        sumload = objective.sumload(os_ms, parameters, convert)  # 总负荷
        obj_record.append({"完工时间": makespan, "最大机器负荷": maxload, "机器总负荷": sumload})
    print('非支配解个数', len(best_list))  # 帕累托前沿个体
    print('非支配解目标值', obj_record)  # 每个个体对应的目标函数值
    return obj_record, best_list


def write_to_file(filename, obj_record, best_list):
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    with open(filename, 'begin') as f:
        f.write(str(now) + '\n')
        f.write(str(obj_record) + '\n')
        for i in range(len(best_list)):
            f.write(str(best_list[i]) + '\n')
        f.write('运行时间' + str(end - start) + '\n')
        f.write('\n')
        f.close()

def find_best(obj_record):
    minimize = float("Inf")
    index = None
    for i in range(len(obj_record)):
        time = obj_record[i]["完工时间"][1]
        if time < minimize:
            index = i
            minimize = time
    return index

# 主函数运行
if __name__ == '__main__':
    patch = r'C:\Users\DELL\PycharmProjects\fjsp_Converttime_Fuzzytime\taobao\Benchmark\Mk05.fjs_fuzzy'
    parameters = readtext.Readtext(patch).readtext()
    convert = covertmatrics.convettime(parameters)

    # 记录开始时间
    start = time.time()
    best_obj, best_list, convergence = main(parameters, convert)
    # 确定前沿pareto数量，及其目标函数
    pareto_obj, pareto_solution = check(best_obj, best_list)
    obj_record, best_list = cal_fuzzy_time(pareto_solution, parameters, convert)
    best_idx = find_best(obj_record)
    # 记录结束时间
    end = time.time()
    print(f'MA运行时间:{end - start}s')

    # 保存目标函数值
    with open(r'./Data/data_MA.pkl', "wb") as f:
        pickle.dump(convergence, f)

    # 保存运行结果
    filename = 'MA运行结果记录.txt'
    write_to_file(filename, obj_record, best_list)

    # 绘制甘特图
    gantt_data = decoding.translate_decoded_to_gantt(
        decoding.decode(parameters, best_list[best_idx][0], best_list[best_idx][1], convert))
    title = "Flexible Job Shop Solution (MA)"  # 甘特图title
    gantt.draw_chart_MA(gantt_data, title, 15)  # 调节数字可以更改图片中标题字体

    # 完工时间
    plt.plot(convergence["makespan"])
    plt.xlabel("Iteration")
    plt.ylabel("$C_{max}$")
    plt.title("Makespan Convergence with Iteration (MA)")
    plt.savefig(r'./PictureSave/makespan_MA.jpg', dpi=400)
    plt.show()
    # 最大负荷
    plt.plot(convergence["maxload"])
    plt.xlabel("Iteration")
    plt.ylabel("Maxworkload")
    plt.title("Maximum workload Convergence with Iteration (MA)")
    plt.savefig(r'./PictureSave/Maxworkload_MA.jpg', dpi=400)
    plt.show()
    # 总负荷
    plt.plot(convergence["sumload"])
    plt.xlabel("Iteration")
    plt.ylabel("Total workload")
    plt.title("Total workload Convergence with Iteration (MA)")
    plt.savefig(r'./PictureSave/TotalWorkload_MA.jpg', dpi=400)
    plt.show()
