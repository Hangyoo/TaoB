# -*- coding: utf-8 -*-
"""demo.py"""
import numpy as np
import geatpy as ea # 导入geatpy库
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from pandas import Series, DataFrame
import sklearn.datasets as datasets
import csv
import pandas as pd  # 用于分析数据集
import matplotlib.pyplot as plt  # 可视化
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
# coding: utf-8
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
from sklearn.svm import SVR
from scipy.optimize import curve_fit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from scipy import integrate
import sys



start_time1 = time.time()

data = []  # 所有数据
Data_feature = []  # 所有数据 输入
Data_target_C = []  # 所有数据 疲劳强度

# 读取CSV数据
csv_file = csv.reader(open('data4.csv'))
for content in csv_file:
    content = list(map(float, content))
    if len(content) != 0:
        data.append(content)
        Data_feature.append(content[0:4])
        Data_target_C.append(content[-1])

print('data=', data)
print('Data_feature=', Data_feature)
print('Data_target_C=', Data_target_C)

data = np.array(data)
data_data = data
Data_feature = np.array(Data_feature)
Data_target_C = np.array(Data_target_C)


ss_data = preprocessing.StandardScaler()
data_nor = ss_data.fit_transform(data)
ss_feature = preprocessing.StandardScaler()
Data_feature_nor = ss_feature.fit_transform(Data_feature)
ss_C = preprocessing.StandardScaler()
Data_target_C_nor = ss_C.fit_transform(Data_target_C.reshape(-1, 1))


train_times = 100

test_size_times = 1  # 迁据集比例次数（0.1-0.8）

for dd in range(test_size_times):

    if dd == 0:
        size_of_train2 = 0.8
    else:
        size_of_train2 = 1

    design_result = np.zeros((train_times, 4))  # 返回来一个给定形状和类型的用0填充的数组 类似于占位符
    best_absorption = np.zeros((train_times, 1))  #   干嘛的
    all = np.zeros((train_times, 5))
    finally_out = np.zeros((30, 1))


    for kk in range(train_times):
        model_A = joblib.load('Results/saved model/Scale_%3.2f Random_%i model_Carbon.pkl' % (size_of_train2, kk))

        """============================目标函数============================"""

        finally_out = np.mat(finally_out)
        def aim(phen):  # 传入种群染色体矩阵解码后的基因表现型矩阵  phen种群 aim(目标函数)定义目标函数 phen传入，x1,x2,x3
            phen = np.column_stack([phen, variable_rest])#和x4拼接
            x = ss_feature.transform(phen)  # 行向量
            y = model_A.NN_predict(x)
            y = ss_C.inverse_transform(y) #列向量 （种群数，1）
            print("y:", y)
            return y.reshape(-1, 1)


        def aimFunc(self, pop):  # 目标函数
            Vars = pop.Phen  # 得到决策变量矩阵
            Vars = np.array(Vars, dtype=int)
            popsize = Vars.shape[0]
            F1 = np.array([float("Inf")] * popsize).reshape(popsize, 1)
            for i in range(popsize):
                x1 = Vars[i, [0]][0]
                x2 = Vars[i, [1]][0]
                x3 = Vars[i, [2]][0]
                f1 = 0

                # 求x4在[37.5,60]的区间下，y的总和
                for x4 in range(37.5, 60.01, 0.01):  # 若要包含x4=60的点，需要将区间设置为60.01
                    X = np.array([x1, x2, x3, x4]).reshape(1, 4)
                    # 计算目标函数值
                    f1 += model_A.NN_predict(X)
                F1[i, 0] = f1
            pop.ObjV = F1  # 把求得的目标函数值赋值给种群pop的ObjV


        """============================变量设置============================?????????????????/"""
        x1 = [0.2, 2]  # 第一个决策变量范围
        x2 = [0.3, 2.7]  # 第二个决策变量范围
        x3 = [0.4, 3.8]


        b1 = [1, 1]  # 第一个决策变量边界，1表示包含范围的边界，0表示不包含
        b2 = [1, 1]  # 第二个决策变量边界，1表示包含范围的边界，0表示不包含
        b3 = [1, 1]


        ranges = np.vstack([x1, x2, x3]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
        borders = np.vstack([b1, b2, b3]).T  # 生成自变量的边界矩阵
        varTypes = np.array([0, 0, 0])  # 3个决策变量x1\x2\x3的类型，0表示连续，1表示离散 0表示染色体解码之后对应的决策变量是离散的，1表示染色体解码之后为连续的

        """==========================染色体编码设置========================="""
        Encoding = 'BG'  # 'BG'表示采用二进制/格雷编码
        codes = [1, 1, 1]  # 决策变量的编码方式，两个1表示变量均使用格雷编码
        precisions = [1, 1, 1]  # 决策变量的编码精度，
        scales = [0, 0, 0]  # 0表示采用算术刻度，1表示采用对数刻度
        # 调用函数创建译码矩阵
        FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)
        """=========================遗传算法参数设置========================"""
        NIND = 30  # 种群个体数目20
        MAXGEN = 10  # 最大遗传代数
        maxormins = [-1]  # 表示目标函数是最小化，元素为-1则表示对应的目标函数是最大化
        selectStyle = 'sus'  # 采用随机抽样选择
        recStyle = 'xovdp'  # 采用两点交叉 在个体编码串中随机设置了两个交叉点，然后再进行部分基因交换
        mutStyle = 'mutbin'  # 采用二进制染色体的变异算子
        pc = 0.9  # 交叉概率0.9
        pm = 0.3 # 整条染色体的变异概率（每一位的变异概率=pm/染色体长度）1
        Lind = int(np.sum(FieldD[0, :]))  # 计算染色体长度
        obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
        var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体



        """=========================开始遗传算法进化========================"""

        start_time = time.time()  # 开始计时

        for x4 in range(37, 60, 1):
            a = [x4]  # 固定参数一维 #一列
            variable_rest = np.tile(a,(NIND,1))#x4变成矩阵（种群数，1）30,1
            Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
            variable = ea.bs2real(Chrom, FieldD)  # 对初始种群进行解码
            print("variable:", variable)
            ObjV = aim(variable)  # 计算初始种群个体的目标函数值
            ObjV_mat = np.mat(ObjV)  # 数组转为矩阵
            print("ObjV:", ObjV)

            best_ind = np.argmin(-finally_out)  # 计算当代最优个体的序号


        # # 开始进化 编码 从表现型到基因型 解码 基因型到表现型
        for gen in range(MAXGEN):
            finally_out = finally_out + ObjV_mat
            print("finally_out:", finally_out, type(finally_out))
            finally_out = np.array(finally_out)
            FitnV = ea.ranking(maxormins * finally_out)  # 根据目标函数大小分配适应度值


            SelCh = Chrom[ea.selecting(selectStyle, FitnV, NIND - 1), :]  # 选择
            SelCh = ea.recombin(recStyle, SelCh, pc)  # 重组（交叉）
            SelCh = ea.mutate(mutStyle, Encoding, SelCh, pm)  # 变异

            # 把父代精英个体与子代的染色体进行合并，得到新一代种群
            Chrom = np.vstack([Chrom[best_ind, :], SelCh])
            Phen = ea.bs2real(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
            finally_out = aim(Phen)  # 求种群个体的目标函数值(objective value)
            # 记录
            best_ind = np.argmin(-finally_out)  # 计算当代最优个体的序号np.argmin():给出最小值的下标,因为选出的最优个体要具有最大目标值,即-ObjV最小
            obj_trace[gen, 0] = np.sum(finally_out) / finally_out.shape[0]   # 记录当代种群的目标函数均值
            obj_trace[gen, 1] = finally_out[best_ind]  # 记录当代种群最优个体目标函数值   gen:所循环到的遗传代数
            var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体(即性能最优所对应的成分工艺参数)
            # 进化完成
            end_time = time.time()  # 结束计时
        ''''''
        # ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像


        """============================输出结果============================"""
        best_gen = np.argmin(-obj_trace[:, [1]])  # 计算当代最优个体的序号
        print('最优解的目标函数值：', obj_trace[best_gen, 1])

        variable = ea.bs2real(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
        b = [a]
        b = np.array(b)
        b = list(b)
        variable_best = np.column_stack([variable, b])

        print('最优解的决策变量值为：')
        for i in range(variable.shape[1]):
            print('x' + str(i) + '=', variable[0, i])
        print('用时：', end_time - start_time, '秒')

        design_result[kk, :] = variable_best
        best_absorption[kk, :] = obj_trace[best_gen, 1]
        all[kk, 0:4] = variable_best
        all[kk, 4] = obj_trace[best_gen, 1]

all_input = ea.bs2real(Chrom[:, :], FieldD)
# all_input = np.column_stack([variable_rest, all_input_nor])  # 在列维度上进行拼接
print("all_input:", all_input.shape)

all_output = finally_out
print("all_output:", all_output)

all_data = np.column_stack([all_input, all_output])
print("all_data:", all_data.shape)

data_all_data= pd.DataFrame(all_data)

data1 = pd.DataFrame(all_input)
data1.to_csv('GA result/%10.2f design result.csv' % (size_of_train2))
data1 = pd.DataFrame(all_output)
data1.to_csv('GA result/%10.2f best fatigue.csv' % (size_of_train2))
data1 = pd.DataFrame(data_all_data)
data1.to_csv('GA result/%10.2f all.csv' % (size_of_train2))
print("#######################################################################")
print("                        ")

