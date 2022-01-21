import numpy as np
import random
from main import test_fun,c_rate,m_rate

'''
交叉变异算子: 模拟二进制交叉，多项式变异
'''

idv = -1

def cross(p1, p2):
    # 模拟二进制交叉
    for i in range(len(p1.X)):
        if np.random.rand() < c_rate:
            p2 = idv.reset_one(p2)
            r1 = 0.7
            r2 = 1 - r1
            x1 = r1 * p1.X[i] + r2 * p2.X[i]
            x2 = r2 * p1.X[i] + r1 * p2.X[i]
            # 越界修复
            if x1 < test_fun.bound[i][0] or x1 > test_fun.bound[i][0]:
                x1 = random.uniform(test_fun.bound[i][0],test_fun.bound[i][1])
            if x2 < test_fun.bound[i][0] or x2 > test_fun.bound[i][0]:
                x2 = random.uniform(test_fun.bound[i][0],test_fun.bound[i][1])
            p1.X[i] = x1
            p2.X[i] = x2
        p2.F_value = test_fun.Func(p2.X)
        p1.F_value = test_fun.Func(p1.X)
    return p2


def mutate(p):

    # 多项式变异
    eta_m = 0.7
    for j in range(len(p.X)):
        r = random.random()
        # 对个体某变量进行变异
        if r <= m_rate:
            y = p.X[j]
            ylow = test_fun.bound[j][0]  # 下界
            yup = test_fun.bound[j][1]   # 上届
            delta1 = 1.0 * (y - ylow) / (yup - ylow)
            delta2 = 1.0 * (yup - y) / (yup - ylow)
            r = random.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if r <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * r + (1.0 - 2.0 * r) * (xy ** (eta_m + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - r) + 2.0 * (r - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - val ** mut_pow
            y = y + deltaq * (yup - ylow)
            # 越界修复
            y = min(yup, max(y, ylow))
            p.X[j] = y
    p.F_value = test_fun.Func(p.X)
    return p


def select(P):
    # 洗牌产生新P
    new_P = []
    for ip in P:
        if ip.p_rank <= 3:
            new_P.append(ip)
    while len(new_P) != len(P):
        p=idv.creat_one()
        new_P.append(p)
    return P
