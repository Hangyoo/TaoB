from NSGA2 import encoding, genetic, objective
from NSGA2 import config, Non_dominated_sorting,PLOT3D
from NSGA2.record import record
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import time


''' ================= 初始参数设定 ======================'''


# 初始参数设置

def main():
    # 随机生成切线时间并保存到”切线时间.txt“中

    # 读取数据
    data1 = pd.read_excel("./data.xlsx", sheet_name='TN')
    data1 = data1.drop(columns=['Cell_ID'], axis=1)
    data1.set_axis([i for i in range(data1.shape[1])], axis="columns", inplace=True)

    # 读取数据
    data2 = pd.read_excel("./data.xlsx", sheet_name='TP')
    data2 = data2.drop(columns=['Cell_ID'], axis=1)
    data2.set_axis([i for i in range(data2.shape[1])], axis="columns", inplace=True)

    # 读取数据
    data3 = pd.read_excel("./data.xlsx", sheet_name='COST')
    data3 = data3.drop(columns=['Cell_ID'], axis=1)
    data3.set_axis([i for i in range(data3.shape[1])], axis="columns", inplace=True)

    gen = 0
    NIND, MAXGEN, pc, pm = config.popSize, config.maxGen, config.pc, config.pm
    population = encoding.initializePopulation()
    best_list, best_obj = [], []

    convergence = {"TN": [], "TP": [], "COST": []}
    before_value = [1e10, 1e10, 1e10]

    while gen < MAXGEN:
        print('第%d次迭代' % gen)
        offspring = genetic.crossover(population)
        offspring = genetic.mutation(offspring)
        Chrom = population + offspring  # (变成适合两倍种群的)
        chroms_obj_record = {}

        for i in range(NIND * 2):
            f1, f2, f3 = objective.aimFunc(Chrom[i],data1,data2,data3)
            chroms_obj_record[i] = [f1, f2, f3]

        # 记录每代最小目标值
        convergence, before_value = record(chroms_obj_record, convergence, before_value)

        front = Non_dominated_sorting.Non_donminated_sorting(NIND, chroms_obj_record)
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


def write_to_file(filename, obj_record, best_list,start,end):
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    with open(filename, 'a') as f:
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

    # 记录开始时间
    start = time.time()
    best_obj, best_list, convergence = main()
    # 确定前沿pareto数量，及其目标函数
    pareto_obj, pareto_solution = check(best_obj, best_list)

    # 记录结束时间
    end = time.time()
    print(f'NSGAII运行时间:{end - start}s')

    # 保存运行结果
    filename = 'NSGAII运行结果记录.txt'
    write_to_file(filename, pareto_obj, pareto_solution,start,end)

    PLOT3D.plot(pareto_obj)

    # 完工时间
    plt.plot(convergence["TN"])
    plt.xlabel("Iteration")
    plt.ylabel("TN")
    plt.title("TN Convergence with Iteration (NSGA-II)")
    plt.savefig(r'./TN_NSGAII.jpg', dpi=400)
    plt.show()
    # 最大负荷
    plt.plot(convergence["TP"])
    plt.xlabel("Iteration")
    plt.ylabel("TP")
    plt.title("TP Convergence with Iteration (NSGA-II)")
    plt.savefig(r'./TP_NSGAII.jpg', dpi=400)
    plt.show()
    # 总负荷
    plt.plot(convergence["COST"])
    plt.xlabel("Iteration")
    plt.ylabel("COST")
    plt.title("COST Convergence with Iteration (NSGA-II)")
    plt.savefig(r'./COST_NSGAII.jpg', dpi=400)
    plt.show()
