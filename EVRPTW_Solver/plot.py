import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel(r'C:\Users\Hangyu\PycharmProjects\TaoB\EVRPTW_Solver\data3.xls')
x = data.iloc[:,0]
y = data.iloc[:,1]

# _meta_solution 文件中每一行对应一个cycle
cycle1 = [0,10,13,2,9,0]
cycle2 = [0,11,12,14,4,6,1,3,0]
cycle3 = [0,8,7,0]
cycle4 = [0,15,5,0]
# cycle4 = [0,27,17,18,0]
# cycle5 = [0,7,5,9,4,0]
# cycle6 = [0,13,23,20,21,0]
# cycle7 = [0,24,25,14,15,0]

cycle = [cycle1,cycle2,cycle3,cycle4]  # 绘制整个图形
# cycle = [cycle1]  # 绘制第1个cycle的路径，即第1台车的路径
# cycle = [cycle3]  # 绘制第2个cycle的路径，即第2台车的路径

# ..以此类推 有16个车，那就是cycle = [cycle16]

plt.scatter(x[0],y[0],color='r',marker='*',s=100)
for i in range(len(x)-1):
    plt.scatter(x[i+1],y[i+1],color='g',marker='o',s=40)

i = 0
for item in cycle:
    c = ['red','blue','orange','green'][i]
    for j in range(len(item)-1):
        plt.plot([x[item[j]],x[item[j+1]]],[y[item[j]],y[item[j+1]]],color=c)
    i+=1

plt.title("Route of problem 3")
plt.xlabel('x')
plt.ylabel('y')
plt.show()