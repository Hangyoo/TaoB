import pickle
import matplotlib.pyplot as plt

# 绘图设置(显示中文)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载PSO算法运行保存的文件
pso_path = r'C:\Users\DELL\PycharmProjects\TB\JSP_1_obj_20210331\result\psowv01.pkl'
with open(pso_path, 'rb') as f:
    data_pso = pickle.load(f)

# 加载GA算法运行保存的文件
ga_path = r'C:\Users\DELL\PycharmProjects\TB\JSP_1_obj_20210331\result\gawv01.pkl'
with open(ga_path, 'rb') as f:
    data_ga = pickle.load(f)

plt.plot(data_pso,'r-')
plt.plot(data_ga,'end-.')

plt.xlabel('迭代次数')
plt.ylabel('完工时间')
plt.legend(["PSO", "GA"])
plt.title("算例 " + pso_path[-8:-4])
plt.savefig(ga_path[-8:-4],dpi=600)
#plt.show()