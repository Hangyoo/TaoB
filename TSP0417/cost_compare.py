import pickle
import matplotlib.pyplot as plt

# 绘图设置(显示中文)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载NSGAII算法运行保存的文件
pso_path = 'cost_nsga2.pkl'
with open(pso_path, 'rb') as f:
    data_before = pickle.load(f)

# 加载改进NSGAII算法运行保存的文件
ga_path = 'cost_nsga2_improve.pkl'
with open(ga_path, 'rb') as f:
    data_after = pickle.load(f)

plt.plot(data_before,'r-')
plt.plot(data_after,'end-.')

plt.xlabel('迭代次数')
plt.ylabel('成本')
plt.legend(["NSGAII", "NSGAII-improve"])
plt.savefig('distance',dpi=600)
plt.show()