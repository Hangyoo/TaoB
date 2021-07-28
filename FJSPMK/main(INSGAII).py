from FJSPMK import encoding, objective, readtext, decoding
from FJSPMK import genetic, config, covertmatrics
from FJSPMK import Non_dominated_sorting, gantt
import matplotlib.pyplot as plt
import pickle
import time

from FJSPMK.record import record

""" ================= 传统FJSP + 切线时间 + 模糊加工时间 ======================"""


# 初始参数设置在config文件中设置

def main(parameters,MC, MD):
    # 随机生成切线时间并保存到”切线时间.txt“中
    gen = 0
    NIND, MAXGEN, pc, pm = config.popSize, config.maxGen, config.pc, config.pm
    population = encoding.initializePopulation(parameters)
    best_list, best_obj = [], []

    convergence = {"makespan": [], "maxload": [], "sumload": [], "cost": []}
    before_value = [1e10, 1e10, 1e10, 1e10]

    while gen < MAXGEN:
        print('第%d次迭代' % gen)
        offspring_crossover = genetic.crossover(population, parameters)
        offspring_mutation = genetic.mutation(offspring_crossover, parameters)
        Chrom = population + offspring_crossover + offspring_mutation
        chroms_obj_record = {}

        for i in range(NIND * 3):
            os_ms = Chrom[i]
            makespan = objective.TimeTakenBenchmark(os_ms, parameters)  # 最大完工时间
            maxload = objective.maxloadBenchmark(os_ms, parameters)  # 最大负荷
            sumload = objective.sumloadBenchmark(os_ms, parameters)  # 总负荷
            sumcost = objective.cost(os_ms, parameters, MC, MD)  # 成本
            chroms_obj_record[i] = [makespan, maxload, sumload, sumcost]

        # 记录每代最小目标值
        convergence, before_value = record(chroms_obj_record, convergence, before_value)

        front = Non_dominated_sorting.Non_donminated_sorting_MA(NIND, chroms_obj_record)
        population, new_popindex = Non_dominated_sorting.Selection(NIND, front, chroms_obj_record, Chrom)

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
    print('非支配解个数', len(list1))  # 帕累托前沿个体
    print('非支配解目标值', list0)  # 每个个体对应的目标函数值
    return list0, list1


def write_to_file(filename, pareto_obj, best_list):
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    with open(filename, 'begin') as f:
        f.write(str(now) + '\n')
        f.write(str(pareto_obj) + '\n')
        for i in range(len(pareto_obj)):
            f.write(str(best_list[i]) + '\n')
        f.write('运行时间' + str(end - start) + '\n')
        f.write('\n')
        f.close()


# 主函数运行
if __name__ == '__main__':
    patch = r'C:\Users\Hangyu\PycharmProjects\TaoB\FJSPMK\Benchmark\8_8.fjs'
    parameters = readtext.Readtext(patch).readtext()
    # MC 原材料费用 [1,2,3,5,9,..] （等于工件个数）
    # MD 机器单位费率  [12,51,48，..] （等于机器个数）
    MC = [1,2,5,4,3,2,1,8]
    MD = [12,11,8,9,10,13,6,7]
    # 记录开始时间
    start = time.time()
    best_obj, best_list, convergence = main(parameters, MC, MD)
    # 确定前沿pareto数量，及其目标函数
    pareto_obj, best_list = check(best_obj, best_list)
    # 记录结束时间
    end = time.time()
    print(f'MA运行时间:{end - start}s')

    # 保存目标函数值
    with open(r'Data/data_INSGAII.pkl', "wb") as f:
        pickle.dump(convergence, f)

    # 保存运行结果
    filename = 'INSGAII运行结果记录.txt'
    write_to_file(filename, pareto_obj, best_list)

    # 绘制甘特图
    gantt_data = decoding.translate_decoded_to_gantt(
        decoding.decodeBenchmark(parameters, best_list[0][0], best_list[0][1]))
    title = "Flexible Job Shop Solution Processing Time (MA)"  # 甘特图title (英文)
    # title = "考虑模糊加工时间的FJSP"    # 甘特图title（中文）
    gantt.draw_chart(gantt_data, title, 15)  # 调节数字可以更改图片中标题字体

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
    # 总价格
    plt.plot(convergence["cost"])
    plt.xlabel("Iteration")
    plt.ylabel("Total cost")
    plt.title("Total cost Convergence with Iteration (MA)")
    plt.savefig(r'./PictureSave/TotalCost_MA.jpg', dpi=400)
    plt.show()

    # 绘制非支配解的在各个目标上的目标值
    for i in pareto_obj:
        plt.plot([1,2,3,4],i)
    plt.title("Objective of Pareto solutions")
    plt.show()
