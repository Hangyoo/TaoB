import geatpy as ea
from nsga2_2_obj_20210129.problem import MyProblem

if __name__ == '__main__':
        # 实例化问题对象
    problem = MyProblem()
    # 构建算法 MOEA/D
    # algorithm = ea.moea_MOEAD_templet(problem,
    #                                  ea.Population(Encoding='RI', NIND=40),
    #                                  MAXGEN=25,  # 最大进化代数。
    #                                  logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
    #                                  trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
    #                                  maxTrappedCount=10)  # 进化停滞计数器最大上限值。

    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding='RI', NIND=100),
                                      MAXGEN=50,  # 最大进化代数。
                                      logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                      trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
                                      maxTrappedCount=20)  # 进化停滞计数器最大上限值。

    # 求解
    res = ea.optimize(algorithm, verbose=True, drawing=2, outputMsg=True, drawLog=False, saveFlag=True)

    print(res)