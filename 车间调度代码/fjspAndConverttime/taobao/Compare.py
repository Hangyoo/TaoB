import pickle
import matplotlib.pyplot as plt

# 读取NAGAII算法目标函数值
with open(r'./Data/data_NSGAII.pkl', "rb") as f:
    convergence_NSGAII = pickle.load(f)

# 读取MA算法目标函数值
with open(r'./Data/data_MA.pkl', "rb") as f:
    convergence_MA = pickle.load(f)

# -----完工时间----
plt.plot(convergence_NSGAII["makespan"],'o-')
plt.plot(convergence_MA["makespan"],'*-.')
plt.xlabel("Iteration")
plt.ylabel("$C_{max}$")
plt.legend(["NSGA-II","MA"])
plt.title("Makespan Convergence with Iteration")
plt.savefig(r'./PictureSave/Compare_makespan.jpg', dpi=400)
plt.show()

# -----最大负荷-----

plt.plot(convergence_NSGAII["maxload"],'o-')
plt.plot(convergence_MA["maxload"],'*-.')
plt.xlabel("Iteration")
plt.ylabel("Maxworkload")
plt.legend(["NSGA-II","MA"])
plt.title("Maximum workload Convergence with Iteration (NSGAII)")
plt.savefig(r'./PictureSave/Compare_Maxworkload.jpg', dpi=400)
plt.show()

# -----总负荷-----
plt.plot(convergence_NSGAII["sumload"],'o-')
plt.plot(convergence_MA["sumload"],'*-.')
plt.xlabel("Iteration")
plt.ylabel("Total workload")
plt.legend(["NSGA-II","MA"])
plt.title("Total workload Convergence with Iteration (NSGAII)")
plt.savefig(r'./PictureSave/Compare_TotalWorkload.jpg', dpi=400)
plt.show()
