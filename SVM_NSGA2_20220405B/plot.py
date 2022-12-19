import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

obj = pd.read_csv(r'C:\Users\Hangyu\PycharmProjects\TaoB\SVM_NSGA2_20220405B\Objective.csv',header=None)
f1 = obj.iloc[:,0]
f2 = obj.iloc[:,1]

plt.scatter(f1,f2)
plt.xlabel("$T_{avg}$",fontsize=13)
plt.ylabel("$T_{rip}$",fontsize=13)
plt.show()