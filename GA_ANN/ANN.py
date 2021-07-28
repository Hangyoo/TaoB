# -*- coding: UTF-8 -*-
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
#数据预处理
data = pd.read_excel(r"C:\Users\Hangyu\Desktop\混凝土抗压强度数据整理.xlsx")
data_inputs = np.array(data.iloc[:,0:8])
data_outputs = np.array(data.iloc[:,8]).reshape(-1,1)
# 数据标准化
mm = MinMaxScaler()
labels = mm.fit_transform(data_outputs)
# 划分测试集和验证集
x_train, x_test, y_train, y_test = train_test_split(data_inputs,labels,random_state=24)

f = open("weights_200_iterations_10%_mutation.pkl","rb")
Coef = pickle.load(f)[0]

class MLPClassifierOverride(MLPRegressor):

    def _init_coef(self, fan_in, fan_out):
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        if fan_in == 8 and fan_out == 200:
            coef_init = Coef[0]
            intercept_init = np.random.uniform(-init_bound, init_bound,200)

        elif fan_in == 200 and fan_out == 100:
            coef_init = Coef[1]
            intercept_init = np.random.uniform(-init_bound, init_bound,100)

        elif fan_in == 100 and fan_out == 1:
            coef_init = Coef[2]
            intercept_init = np.random.uniform(-init_bound, init_bound,1)

        else:
            sys.exit("ANNet文件中神经元个数输入有误！")

        return coef_init, intercept_init


rf = MLPClassifierOverride(hidden_layer_sizes=(200,100),learning_rate_init= 0.001,activation='logistic',solver='adam', alpha=1e-8)  # 神经网络
# 训练分类器
rf=rf.fit(x_train, y_train)


# 在训练集上测试
y_pred_train = rf.NN_predict(x_train)
r = r2_score(y_train,y_pred_train)
mse = mean_squared_error(y_train,y_pred_train)
print("ANN训练集R2:",r)
print("ANN训练集MSE:",mse)

# 在预测集上测试
y_pred_test = rf.NN_predict(x_test)
r = r2_score(y_test,y_pred_test)
mse = mean_squared_error(y_test,y_pred_test)
print("_____________")
print("ANN训练集R2:",r)
print("ANN训练集MSE:",mse)

# 预测值(反归一化)
# y_pred = mm.inverse_transform(y_pred_test.reshape(-1,1))
# print(y_pred)

# 绘制优化前及优化后的指标差异
model = MLPRegressor(hidden_layer_sizes=(200,100),learning_rate_init= 0.001,activation='logistic',solver='adam', alpha=1e-8)  # 神经网络
model=model.fit(x_train, y_train)


plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.plot(rf.loss_curve_,"r")
plt.plot(model.loss_curve_,"end")
plt.legend(["ANN+GA","ANN"])
plt.show()


