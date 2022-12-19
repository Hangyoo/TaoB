from nsga2_2_obj_20220213.gmdhpy.gmdh import Regressor
import pandas as pd
from sklearn import preprocessing
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")



df=pd.read_excel("data.xlsx",header=0,skiprows=[1],usecols=range(2,12),dtype=float)

scaler_x=preprocessing.StandardScaler()
train_x = scaler_x.fit_transform(df[df.columns[:-1]])

scaler_y=preprocessing.StandardScaler()
train_y = scaler_y.fit_transform(df[[df.columns[-1]]])

ref_functions=['quadratic']

if __name__ == "__main__":
    model = Regressor(ref_functions=ref_functions[0],
                      criterion_type='validate',
                      seq_type = 'mode3_1',#数据集划分
                      max_layer_count = 11,#最大层数
                      feature_names=df.columns[:-1],
                      criterion_minimum_width=5,
                      stop_train_epsilon_condition=0.001,#停止训练阈值
                      layer_err_criterion='top',#层误差计算标准
                      l2=0.5,#正则化值
                      n_jobs='max')#训练模型的并行进程（线程）的数量，默认为1
    model.fit(train_x, train_y)

    # 模型保存
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # y_pred = model.predict(train_x)
    # print(train_x)
    # X = np.array([10, 20, 30, 40, 25, 6, 7, 8, 9]).reshape(1, 9)
    # # 数据处理
    # x = scaler_x.transform(X)[0]
    # print(x)
    # y_pred = scaler_y.inverse_transform(y_pred)
    # print(y_pred)