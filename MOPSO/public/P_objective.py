import random

import numpy as np


def P_objective(Operation,Problem,M,Input):
    [Output, Boundary, Coding] = P_DTLZ(Operation, Problem, M, Input)
    if Boundary == []:
        return Output
    else:
        return Output, Boundary, Coding

def P_DTLZ(Operation,Problem,M,Input):
    Boundary = []
    Coding = ""
    k = 1
    K = [5, 10, 10, 10, 10, 10, 20]
    K_select = K[k - 1]
    if Operation == "init":
        D = M + K_select - 1
        MaxValue = np.ones((1, D))
        MinValue = np.zeros((1, D))
        Population = np.random.random((Input, D))
        Population = np.multiply(Population, np.tile(MaxValue, (Input, 1))) +\
            np.multiply((1-Population), np.tile(MinValue, (Input, 1)))
        Boundary = np.vstack((MaxValue, MinValue))
        Coding = "Real"
        return Population, Boundary, Coding

    elif Operation == "value":
        D = 2
        MaxValue = np.array([2280, 11.3]).reshape((1, D))
        MinValue = np.array([760, 4.8]).reshape((1, D))
        Boundary = np.vstack((MaxValue, MinValue))
        Population = np.random.random((100, D))
        for i in range(100):
            Population[i, 0] = MinValue[0][0] + random.random()*(MaxValue[0][0]-MinValue[0][0])
            Population[i, 1] =MinValue[0][1] + random.random()*(MaxValue[0][1]-MinValue[0][1])
        FunctionValue = np.zeros((Population.shape[0], M))

        if Problem == "Problem":
            print("执行")

            V = 53470.0
            O = 0.2
            alpha = 71 * np.pi / 180
            a = 42.22
            H = 30.0
            T1 = 0.17
            x1 = Population[:, 0]
            x2 = Population[:, 1]
            b = (2.6725 + 0.001082 * x1 - 0.09283 * x2 + 0.001292 * x2 * x2 - 0.00001 * x1 * x2) ** 2
            h = (1.1942 + 0.000297 * x1 - 0.0055 * x2 + 0.000262 * x2 * x2 - 0.000008 * x1 * x2) ** 2
            FunctionValue[:,0] = V / h / b / (1 - O) / x2 + a * T1 * H / h / b / (1 - O)
            FunctionValue[:,1] = (h ** 2) / 2 / np.tan(alpha)

        return FunctionValue, Boundary, Coding








