import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # ⽣成多项式⽤的
import matplotlib.pyplot as plt

# 处理数据中的缺失值
def handle_gold(data_path):
    data = pd.read_csv(data_path)
    # 处理空缺值，设为平均
    # data['USD (PM)'] = data['USD (PM)'].fillna(data['USD (PM)'].median())
    # ⽤缺失值上⾯的值替换缺失值
    data = data.fillna(axis=0, method='ffill')
    data.to_csv(r'C:\Users\Hangyu\PycharmProjects\TaoB\SVR_stock/new_data.csv')

# 绘图
def show_gold(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 0]
    y_data = data.iloc[1:, 1]
    plt.scatter(x_data, y_data, s=1)
    plt.show()

# 建立模型
def build_model(data_path):
    data = pd.read_csv(data_path)
    x_data = data.iloc[1:, 0]
    y_data = data.iloc[1:, 2]
    # 转换为⼆维数据
    # degree = n，相当于n次⽅拟合
    poly = PolynomialFeatures(degree=6)
    # 特征处理
    x_data = np.array(x_data).reshape((len(x_data), 1))
    x_poly = poly.fit_transform(x_data)
    # Linear模型拟合
    model = LinearRegression()
    model.fit(x_poly, y_data)
    print('系数：', model.coef_)
    print('截距：', model.intercept_)

    # SVR模型拟合
    model1 = SVR()
    model1.fit(x_poly, y_data)

    # 两个模型分别预测 并 画图
    plt.scatter(x_data, y_data, color='g',s=1,label='True data')
    plt.plot([i for i in range(len(x_data))], model.predict(x_poly), 'r',label='Predict data (Linear)')
    plt.plot([i for i in range(len(x_data))], model1.predict(x_poly), 'b',label='Predict data(SVM)')
    plt.title('Polynomial Regression Stock Model')
    plt.xlabel('days')
    plt.ylabel('dollars$')
    plt.legend()
    plt.show()
    return len(y_data), model.coef_[1:], model.intercept_

if __name__ == '__main__':
    data_path = r"C:\Users\Hangyu\PycharmProjects\TaoB\SVR_stock\gold_data.csv"
    handle_gold(data_path)
    show_gold(data_path)
    new_data = r"C:\Users\Hangyu\PycharmProjects\TaoB\SVR_stock\new_data.csv"
    build_model(new_data)
