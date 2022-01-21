'''
求解问题部分
'''

from green_optimization.nsga_ii_new import function1, function2
import numpy as np

"""
最小化目标双目标优化问题
min f1 
min f2 

s.t.
0.01<= x1 <= 0.3
0.01<= x2 <= 0.3
0.01<= x3 <= 0.3
0.01<= x4 <= 0.8
0.01<= x5 <= 0.4
0.01<= x6 <= 0.4
0.01<= x7 <= 0.25
0.01<= x8 <= 0.25
"""

dimention = 8
bound = [[0.01, 0.3],[0.01, 0.3],[0.01, 0.3],[0.01, 0.8],[0.01, 0.4],[0.01, 0.4],[0.01, 0.25],[0.01, 0.25]]


def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]

def F1(X):
    return function1(X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7])


def F2(X):
    return function2(X[0],X[1],X[2],X[3],X[4],X[5],X[6],X[7])
