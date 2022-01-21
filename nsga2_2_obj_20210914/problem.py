import numpy as np
import pandas as pd
import geatpy as ea

"""
最小化目标双目标优化问题
min Cmg + Cbuy
min Cen

"""

# 电价
price = [0.191,0.191,0.191,0.191,0.191,0.191,0.511,0.511,0.511,1.001,1.001,1.001,1.001,\
         1.001,0.511,0.511,0.511,1.001,1.001,1.001,0.511,0.511,0.191,0.191]

# 风电预测数据
PV = [0,0,0,0,0,0,12.81254015,266.3208738,542.3182246,622.6416713,410.5500034,494.4705134,551.5484107,474.3227225,\
      467.1337374,284.8276747,157.6329897,53.79625388,0.692080853,0,0,0,0,0]

# 典型日负荷
load = [546.7625,512.7625,483.7,461.6375,446.525,436.075,430.4375,428.7875,434.425,457.95,487.375,518.65,550.775,\
        581.275,611.6625,637.9125,657.4,680.375,690.05,689.0375,674.175,652.75,625.1125,586.6625]

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M=2):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 24  # 初始化Dim（决策变量维数）
        maxormins = [1,1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [200] * Dim # 决策变量下界
        ub = [600]* Dim  # 决策变量上界
        lbin = [1]* Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]* Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Vars = np.array(Vars, dtype=int)

        a = 0.252 # 单位电力所需成本系数
        b = 0.412  # 单位电力所需管理成本系数
        c = (724/1000)*0.047 + (0.0036/1000)*4.052 + (0.2/1000)*9.132  # 单位电力所需环境治理成本系数 (转换为kg * 处理成本)


        soc_min = 0.15
        soc_max = 0.95
        coe_min = 600 * soc_min
        coe_max = 600 * soc_max
        cap_min = 180
        cap_max = 200

        f1 = [] # 记录成本
        f2 = [] # 记录环境保护成本
        for pop_idx in range(len(Vars)):
            f1_temp = 0
            f2_temp = 0
            coe = 600 * 0.9
            for i in range(24):
                x = Vars[pop_idx, [i]]
                gap = load[i] - PV[i] # 差额
                gap -= x
                f2_temp += np.abs(gap) * c
                if gap > 0:
                    if (coe - coe_min) > gap:
                        # 判断使用电池或使用柴电
                        p1 = gap * price[i]
                        p2 = gap * (a + b)
                        if p1 < p2:
                            coe -= gap
                            f1_temp += p1
                        else:
                            f1_temp += p2
                    else:
                        # 柴电发电量
                        gap -= (coe - cap_min)
                        p2 = gap * (a + b)
                        f1_temp += p2
                if gap < 0:
                    # 给电池充电
                    # 超出电池容量
                    if (coe + gap) > coe_max:
                        if np.abs(gap) > cap_max:
                            #f1_temp -= cap_max * price[i]
                            coe += coe_max
                        else:
                            #f1_temp -= (coe_max-coe)*price[i]
                            coe = coe_max
                    else: # 未超出电池容量
                        if np.abs(gap) > cap_max:
                            #f1_temp -= cap_max * price[i]
                            coe += coe_max
                        else:
                            coe += gap
                            #f1_temp -= gap*price[i]

            f1.append(f1_temp)
            f2.append(f2_temp)
        f1 = np.array(f1).reshape(len(Vars),1)
        f2 = np.array(f2).reshape(len(Vars), 1)


        pop.ObjV = np.hstack([f1, f2])  # 把求得的目标函数值赋值给种群pop的ObjV
