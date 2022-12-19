import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def getData(path):
    '''对数据进行预处理，将字符串批量转换为频率'''
    with open(path,'r') as f:
        data = []
        features = []
        i = 1
        for line in f.readlines():
            if i % 2 == 0:
                data.append(line.split('\n')[0])
            i += 1

    # 统计词频
    for item in data:
        dir = {'A': 0, 'E': 0, 'F': 0, 'P': 0, 'D': 0, 'Y': 0, 'S': 0, 'Q': 0, 'M': 0, 'G': 0, 'H': 0, 'K': 0, 'R': 0, 'V': 0, 'L': 0, 'T': 0, 'N': 0, 'I': 0, 'C': 0, 'W': 0}
        length = len(item)
        for char in item:
            dir[char] += 1
        feature = [round(num / length,8) for num in list(dir.values())]
        features.append(feature)

    features = np.array(features).reshape(len(data),20)

    return features


# 读取 正 负样本
path_pos = r'C:\Users\Hangyu\PycharmProjects\TaoB\protein_classifier\positive.txt'
path_neg = r'C:\Users\Hangyu\PycharmProjects\TaoB\protein_classifier\negative.txt'

# 赋标签
pos_y = np.array([[1 for i in range(len(getData(path_pos)))]])
neg_y = np.array([[0 for i in range(len(getData(path_neg)))]])
# pos_data = np.insert(getData(path_pos),20,values=pos_y,axis=1)
# neg_data = np.insert(getData(path_neg),20,values=neg_y,axis=1)
pos_data = getData(path_pos)
neg_data = getData(path_neg)

# 划分训练集合预测集
X = np.concatenate([pos_data,neg_data],axis=0)
Y = np.concatenate([pos_y[0],neg_y[0]],axis=0)
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# SVM模型训练
model = SVC()

# 数据拟合
model.fit(X_train, y_train)

# 批量预测测试
score1 = model.score(X_train,y_train)
score2 = model.score(X_test,y_test)

print(f"在训练集上的预测准确率为:{score1}")
print(f"在训练集上的预测准确率为:{score2}")


#######输入自己的数据进行预测######
# 数据地址
path = r'C:\Users\Hangyu\PycharmProjects\TaoB\protein_classifier\negative.txt'
# 数据预处理
data = getData(path)
print(data)
label = model.predict(data)
print('预测标签为:')
print(label)


