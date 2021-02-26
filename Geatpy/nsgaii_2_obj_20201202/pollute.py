from pyswmm import Simulation,Nodes,LidGroups
import numpy as np

def pollute(x1,x2,x3):
    #todo 根据电脑中数据位置更改此处地址
    with Simulation(r"F:\taobao\ii.inp") as sim:
        lid1=LidGroups(sim)['9']
        for x in lid1:
            x.number=x1
        lid2=LidGroups(sim)['8']
        for x in lid2:
            x.number=x2
        lid3=LidGroups(sim)['5']
        for x in lid3:
            x.number=x3
        out=Nodes(sim)['4']
        for i in sim:
            pass
        y=int(out.outfall_statistics['pollutant_loading']['tss'])
    return y

def funtion2(x1,x2,x3):
    x1, x2, x3 = x1.tolist(),x2.tolist(),x3.tolist()
    length = len(x1)
    y = []
    for i in range(length):
        val = pollute(x1[i][0], x2[i][0], x3[i][0])
        y.append(val)
    y = np.array(y).reshape(length, 1)
    return y

if __name__ == "__main__":
    y = pollute(0,0,0)