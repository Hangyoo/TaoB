import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

'''
训练随机深林模型, 查找特征贡献度 (运行该文件)
'''

# 防止警告，报错
warnings.filterwarnings("ignore")

start = time.time()

# 在这里导入数据文件
data = pd.read_excel("data.xlsx",header=None)
data = data.drop(index=0)

# 在这里输出想保留特征的数目
num = 60

# 特征 和 label
X = data.iloc[:,:247]
Y1 = data.iloc[:,247]
Y2 = data.iloc[:,248]
Y3 = data.iloc[:,249]
Y4 = data.iloc[:,250]
Y5 = data.iloc[:,251]
Y6 = data.iloc[:,252]


for i in range(len([Y1,Y2,Y3,Y4,Y5,Y6])):
    Y = [Y1,Y2,Y3,Y4,Y5,Y6][i]
    Y = Y.astype("int")
    # 打乱数据集
    x_train, x_test, y_train, y_text = train_test_split(X, Y, random_state=21, test_size=0.8)

    # 建立随机深林模型
    if i == 0:
        rf = RandomForestRegressor(n_estimators=130)
    else:
        rf = RandomForestClassifier(n_estimators=130)
    rf.fit(x_train,y_train)
    # r2 越大代表模型越好
    r2 = rf.score(x_test,y_text)

    # 提取特征重要度，对特征重要度进行排序，输出前num个重要特征
    featur_importance = [*zip(map(lambda x: round(x,4), rf.feature_importances_),range(247))]
    featur_importance = sorted(featur_importance,reverse=True)
    res = featur_importance[:num]

    print(f'训练得分R2:{r2}')
    print(f"模型重要度排序为(重要度, 列):\n {res}")
    print(f"运行耗时：{time.time() - start}s")

    # bonus 绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar([i for i in range(num)],[i[0] for i in res])
    ax.set_ylabel('Feature importance')
    ax.set_xlabel('Feature')
    plt.show()
