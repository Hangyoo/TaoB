import numpy as np
import random

# 产生基于工序的染色体
def generatechrom(T,NIND):  #T为加工时间矩阵  NIND为初始种群的数量

    PNumber, MNumber = T.shape

    WNumber = PNumber*MNumber
    Chrom = np.zeros((NIND,WNumber))

    # 产生初始种群，以及初始种群中个体的编码
    Number = [0] * PNumber

    for j in range(NIND):

        for i in range(PNumber):
            Number[i] = MNumber  # Number总储存了每个工件的工序数
        Numbertemp = Number

        for i in range(WNumber):

            val = random.randint(1,PNumber)   #随机选一个工作号，从1开始

            while Numbertemp[val-1] == 0:       #如果工作对应的工序数不为0，那么可以安排工作
                val = random.randint(1, PNumber)

            Chrom[j,i] = val
            Numbertemp[val-1] -= 1               #安排一个工序后，就可以给这个工件剩余工序-1

    return  Chrom

def randomSchedule(j, m):
    """
    Returns begin random schedule for j jobs and m machines,
    i.e. begin permutation of 0^m 1^m ... (j-1)^m = (012...(j-1))^m.
    """
    schedule = [i for i in list(range(j)) for m in range(m)]   #生成调度工序
    random.shuffle(schedule)
    return schedule


if __name__ == '__main__':
    a = [[1, 3, 6, 7, 3, 6], [8, 5, 10, 10, 10, 4], [5, 4, 8, 9, 1, 7], [5, 5, 5, 3, 8, 9], [9, 3, 5, 4, 3, 1],
         [3, 3, 9, 10, 4, 1]]
    T = np.array(a).reshape((len(a),len(a[0])))
    Chrom = generatechrom(T,40)
    print(Chrom)
