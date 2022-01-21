#encoding: utf-8
import numpy as np
from Mopso import *
from public import P_objective

 
def main():

    particals = 100  #粒子群的数量
    cycle_ = 100     #迭代次数
    mesh_div = 10    #网格等分数量
    thresh = 300     #外部存档阀值

    Problem = "Problem"
    M = 2 # 目标个数
    Population, Boundary, Coding = P_objective.P_objective("value", Problem, M, particals)
    max_ = Boundary[0]
    min_ = Boundary[1]


    mopso_ = Mopso(particals,max_,min_,thresh,mesh_div) #粒子群实例化
    pareto_in,pareto_fitness = mopso_.done(cycle_) #经过cycle_轮迭代后，pareto边界粒子
    np.savetxt("pareto_in.txt",pareto_in)#保存pareto边界粒子的坐标
    np.savetxt("pareto_fitness.txt",pareto_fitness) #打印pareto边界粒子的适应值

    print ("\n 算法执行完毕!")


 
if __name__ == "__main__":
    main()
