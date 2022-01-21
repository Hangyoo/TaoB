'''
种群个体部分，抽象成类
'''
import Operators as ga


class Individual:
    Np = 0  # 支配当前个体的数量
    Sp = []  # 当前个体支配的个体集（所在种群编号）
    p_rank = 0  # 当前的序
    dp = 0  # 拥挤度
    X = []
    F_value = []
    test_func = ga.test_fun

    def __init__(self):
        pass

    def creat_one(self):
        one = Individual()
        one.Np = 0
        one.Sp = []
        one.p_rank = 0
        one.dp = 0
        one.X = []
        for i in range(self.test_func.dimention):
            one.X.append(ga.test_fun.bound[i][0] + (ga.test_fun.bound[i][1] - ga.test_fun.bound[i][0]) * ga.np.random.rand(1)[0])
        one.X = ga.np.array(one.X)
        one.F_value = ga.test_fun.Func(one.X)
        return one

    def reset_one(self, one):
        one.Np = 0
        one.Sp = []
        one.p_rank = 0
        one.dp = 0
        return one
