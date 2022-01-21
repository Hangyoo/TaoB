
'''---------------------计算目标函数-------------------------------'''


"""
最小化目标双目标优化问题
min f1 = TN
min f2 = TP
min f3 = COST
"""


def aimFunc(individual,data1,data2,data3):
    f1 = 0
    f2 = 0
    f3 = 0
    for j in range(len(individual)):
        gene = individual[j]
        f1 += data1.iloc[j, gene]
        f2 += data2.iloc[j, gene]
        f3 += data3.iloc[j, gene]
    return round(f1,3),round(f2,3),round(f3,3)