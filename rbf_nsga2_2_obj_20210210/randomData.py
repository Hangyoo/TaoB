import random
import numpy as np
import pandas as pd

"""数据随机生成"""

lower = [3.654,0.12,3.654,0.27,0,0.12,0,0.27,0.554,0.006,3.354,0.06]
upper = [4.054,0.2,4.054,0.3,0.4,0.2,0.4,0.3,0.754,0.1,3.554,0.1]

# 随机生成符合要求的x
data = []
for i in range(12):
    temp = []
    for j in range(100):
        a = random.uniform(lower[i], upper[i])
        temp.append(a)
    data.append(temp)

# 生成相互冲突的y
y1 = []
y2 = []
for i in range(100):
    val = 0
    for j in range(12):
        val += data[j][i]**2
    val += data[0][i] ** 5
    y1.append(val)
    y2.append(100/val+20)
data.append(y1)
data.append(y2)


data = np.array(data).T
data = pd.DataFrame(data)
data.to_excel("exampleData.xls",header=False, index=False)



