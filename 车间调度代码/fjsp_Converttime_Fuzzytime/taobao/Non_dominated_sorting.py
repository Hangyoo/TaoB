import numpy as np


'''---------非支非支配排序(多目标)---------'''
def Non_donminated_sorting(NIND, chroms_obj_record):
    # front = {0: [6, 9, 12, 27, 32, 35], 1: [0, 13, 29, 37], 2: [2, 11, 34, 38], 3: [18, 23, 15, 28]...}
    chroms_obj_record = [i for i in chroms_obj_record.values()]  # 转换为列表
    f = np.reshape(chroms_obj_record,(2*NIND,len(chroms_obj_record[0])))
    Rank = np.zeros(2*NIND)  # [0. 2. 1. 1. 1. 0. 0. 0. 2. 1.]
    front = []     # [[0, 5, 6, 7], [2, 3, 4, 9], [1, 8]]
    rank = 0

    n_p = np.zeros(2*NIND)
    s_p = []
    for p in range(2*NIND):
        a = (f[p, :] - f[:, :] <= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] >= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
        n_p[p] = len(temp1)  # p所支配个数
    # 添加第一前沿
    front.append(list(np.where(n_p == 0)[0]))

    while len(front[rank]) != 0:    # 生成其他前沿
        elementset = front[rank]
        n_p[elementset] = float('inf')
        Rank[elementset] = rank
        rank += 1

        for i in elementset:
            temp = s_p[i]
            n_p[temp] -= 1
        front.append(list(np.where(n_p == 0)[0]))
    front.pop()
    return front

def Non_donminated_sorting_MA(NIND, chroms_obj_record):
    chroms_obj_record = [i for i in chroms_obj_record.values()]  # 转换为列表
    f = np.reshape(chroms_obj_record,(3*NIND,len(chroms_obj_record[0])))
    Rank = np.zeros(3*NIND)
    front = []
    rank = 0

    n_p = np.zeros(3*NIND)
    s_p = []
    for p in range(3*NIND):
        a = (f[p, :] - f[:, :] <= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] >= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
        n_p[p] = len(temp1)  # p所支配个数
    # 添加第一前沿
    front.append(list(np.where(n_p == 0)[0]))

    while len(front[rank]) != 0:    # 生成其他前沿
        elementset = front[rank]
        n_p[elementset] = float('inf')
        Rank[elementset] = rank
        rank += 1

        for i in elementset:
            temp = s_p[i]
            n_p[temp] -= 1
        front.append(list(np.where(n_p == 0)[0]))
    front.pop()
    return front


'''--------拥挤距离计算(多目标)---------'''
def Calculate_crowding_distance(rank, chroms_obj_record):
    # rank = [6, 9, 12, 27, 32, 35]  单一等级（是front的一部分）
    # {0: [35016.0, 1753], 1: [29783.0, 1697], 2: [34906.0, 1677], ....2 * NIND - 1: [TWET, makespan]}
    distance = {i: 0 for i in rank}  # i =  前沿中个体序号
    target_num = len(chroms_obj_record[0])
    for o in range(target_num):  # 对各个目标拥挤距离依次进行计算
        obj = {m: chroms_obj_record[m][o] for m in rank}
        sorted_index = sorted(obj, key=obj.get)
        distance[sorted_index[0]] = distance[sorted_index[len(rank) - 1]] = float('inf')  # 边界最大最小为无穷

        for i in range(1, len(rank) - 1):  # 对第二到倒数第二进行循环
            if len(set(obj.values())) == 1:  # 该前沿层只有一个个体
                pass
            else:  # 拥挤距离 = 相邻目标差/最大最小差   之和
                distance[sorted_index[i]] += \
     (obj[sorted_index[i + 1]] - obj[sorted_index[i - 1]]) / (obj[sorted_index[len(rank) - 1]] - obj[sorted_index[0]])
    return distance  # {15:1800,17:1256,..}

'''----------选择生成新种群（多目标）----------'''
def Selection0(NIND, front, chroms_obj_record, chromosome):
    N = 0
    new_popindex = []  # 储存被选中个体的序号
    population = []    # 储存被选中个体
    while N < NIND:
        for i in range(len(front)):  # 前沿个数
            N = N + len(front[i])
            if N > NIND:
                distance = Calculate_crowding_distance(front[i], chroms_obj_record)
                sorted_cdf = sorted(distance, key=distance.get, reverse=True)
                for j in sorted_cdf:
                    if len(new_popindex) == NIND:
                        break
                    new_popindex.append(j)
                break
            else:
                new_popindex.extend(front[i])

    for index in new_popindex:
        population.append(chromosome[index])
    return population, new_popindex  #[[],[],[],]   [15,13,14,12]  种群及种群个体编号

'''----------改进选择生成新种群（多目标）----------'''
def Selection(NIND, front, chroms_obj_record, chromosome):
    N = 0
    new_popindex = []  # 储存被选中个体的序号
    population = []    # 储存被选中个体
    while N < NIND:
        for i in range(len(front)):  # 前沿个数
            num = round(0.6 * len(front[i]))
            N = N + num
            if N > NIND:
                distance = Calculate_crowding_distance(front[i], chroms_obj_record)
                sorted_cdf = sorted(distance, key=distance.get, reverse=True)
                for j in sorted_cdf:
                    if len(new_popindex) == NIND:
                        break
                    new_popindex.append(j)
                break
            else:
                new_popindex.extend(front[i][0:num])

    for index in new_popindex:
        population.append(chromosome[index])
    return population, new_popindex  #[[],[],[],]   [15,13,14,12]  种群及种群个体编号


