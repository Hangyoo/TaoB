import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 风电预测数据
PV = [0,0,0,0,0,0,12.81254015,266.3208738,542.3182246,622.6416713,410.5500034,494.4705134,551.5484107,474.3227225,\
      467.1337374,284.8276747,157.6329897,53.79625388,0.692080853,0,0,0,0,0]

# 典型日负荷
load = [546.7625,512.7625,483.7,461.6375,446.525,436.075,430.4375,428.7875,434.425,457.95,487.375,518.65,550.775,\
        581.275,611.6625,637.9125,657.4,680.375,690.05,689.0375,674.175,652.75,625.1125,586.6625]

# 读取数据
path = r'C:\Users\Hangyu\PycharmProjects\TaoB\nsga2_2_obj_20210914\Result\Chrom.csv'
data = pd.read_csv(path,header=None)
idx = 0 # 选择pareto中第一个解

# 柴电发电量
data = data.iloc[idx,:].tolist()
# 电池用量
data_battery = [data[i] - (load[i]-PV[i]) for i in range(len(data))]

print("光伏发电：\n", [round(i,2) for i in PV])
print("柴电发电：\n", [round(i,2) for i in data])
print("柴电发电(放电+, 储能-)：\n", [round(i,2) for i in data_battery])

plt.plot([i+1 for i in range(24)],load)
plt.plot([i+1 for i in range(24)],PV)
plt.plot([i+1 for i in range(24)],data)
plt.plot([i+1 for i in range(24)],data_battery)
plt.title("含储能装置调度结果")
plt.legend(["用电需求","光伏","柴电","储能"])
plt.show()

plt.plot([i+1 for i in range(24)],data)
plt.plot([i+1 for i in range(24)],data_battery)
plt.title("柴电-储能调度结果")
plt.legend(["柴电","储能"])
plt.show()

