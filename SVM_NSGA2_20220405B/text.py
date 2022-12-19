import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 读取数据
data = pd.read_excel("./data.xls")
f1 = [x[0] for x in np.array(data.iloc[:,5:6])]
f2 = [x[0] for x in np.array(data.iloc[:,6:7])]


# 读取训练好的机器学习模型

with open("Svm_model1.pkl","rb") as F1:
    Svm_model1 = pickle.load(F1)
with open("Svm_model2.pkl", "rb") as F2:
    Svm_model2 = pickle.load(F2)

with open("Bag_model1.pkl","rb") as F1:
    Lin_model1 = pickle.load(F1)
with open("Bag_model2.pkl", "rb") as F2:
    Lin_model2 = pickle.load(F2)


Svm_predict_f1 = []
Svm_predict_f2 = []
for i in range(len(f1)):
    x1,x2,x3,x4,x5 = data.iloc[i,0],data.iloc[i,1],data.iloc[i,2],data.iloc[i,3],data.iloc[i,4]
    X = np.array([x1, x2, x3, x4, x5]).reshape(1, 5)
    Svm_predict_f1.append(Svm_model1.predict(X))
    Svm_predict_f2.append(Svm_model2.predict(X))

Lin_predict_f1 = []
Lin_predict_f2 = []
for i in range(len(f1)):
    x1,x2,x3,x4,x5 = data.iloc[i,0],data.iloc[i,1],data.iloc[i,2],data.iloc[i,3],data.iloc[i,4]
    X = np.array([x1, x2, x3, x4, x5]).reshape(1, 5)
    Lin_predict_f1.append(Lin_model1.predict(X)[0])
    Lin_predict_f2.append(Lin_model2.predict(X)[0])


plt.plot([i for i in range(len(f1))],f1,'o')
plt.plot([i for i in range(len(f1))],Svm_predict_f1,'1')
plt.plot([i for i in range(len(f1))],Lin_predict_f1,'*')
plt.legend(['True value','SVM_Predict','Bagging_Predict'])
plt.title('loss')
plt.show()


plt.plot([i for i in range(len(f2))],f2,'o')
plt.plot([i for i in range(len(f2))],Svm_predict_f2,'1')
plt.plot([i for i in range(len(f2))],Lin_predict_f2,'*')
plt.legend(['True value','SVM_Predict','Bagging_Predict'])
plt.title('power')
plt.show()
