import numpy as np
import random

# 加工不同工件的需要切线时间
def convettime(parameters):
    # ni 为列表，储存每个工件的总工序数
    ni = []
    for job in parameters['jobs']:
        ni.append(len(job))

    matric = [[random.randint(1,10)/10 for _ in range(sum(ni))] for _ in range(sum(ni))]
    matric = np.array(matric)
    matric = np.around(matric,1)

    a = [0]
    for i in range(len(ni)):
        a.append(sum(ni[:i+1]))
        # [0, 3, 5, 7]

    period = []
    for i in range(1, len(a)):
        upper = a[i]
        lower = a[i-1]
        period.append((lower, upper))

    for (lower, upper) in period:
        for i in range(lower, upper):
            for j in range(lower, upper):
                matric[i][j] = 0

    for i in range(sum(ni)):
        for j in range(i+1):
            matric[j][i] = matric[i][j]

    # 保存至txt文件中
    np.savetxt("切线时间.csv", matric)

    return matric

