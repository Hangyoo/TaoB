import pickle
import matplotlib.pyplot as plt

with open('data_ga1.pkl','rb') as f1:
    data1 = pickle.load(f1)
with open('data_ga2.pkl','rb') as f2:
    data2 = pickle.load(f2)

# 绘制迭代图
plt.plot(data1,"r")
plt.plot(data2,"g")
plt.ylabel('Fitness History')
plt.xlabel('Generations')
plt.show()