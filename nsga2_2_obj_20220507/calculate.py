import pandas as pd
import numpy as np

V = pd.read_csv('.\Population Info\Phen.csv',header=None)
V = np.array(V).tolist()
v = []
for item in V:
    v.append(item[0])

A = 168
B = 10.263
v0 = 19
D = 23142

def fun(v):
    val = []
    for item in v:
        res = ((A * (item / v0)**3) * (D/(24*item)) + B*(D/(24*item))) * 3.17
        val.append(res)
    return val


val = fun(v)
np.savetxt('result.csv',val)
print(f'val最大值：{max(val)}')
print(f'val最小值：{min(val)}')