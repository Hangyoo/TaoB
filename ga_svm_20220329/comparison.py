import numpy as np
import matplotlib.pyplot as plt

DE_best = np.loadtxt(r'C:\Users\Hangyu\PycharmProjects\TaoB\ga_svm_20220329\DE_best.txt').tolist()
GA_best = np.loadtxt(r'C:\Users\Hangyu\PycharmProjects\TaoB\ga_svm_20220329\GA_best.txt').tolist()

DE_avg = np.loadtxt(r'C:\Users\Hangyu\PycharmProjects\TaoB\ga_svm_20220329\DE_avg.txt').tolist()
GA_avg = np.loadtxt(r'C:\Users\Hangyu\PycharmProjects\TaoB\ga_svm_20220329\GA_avg.txt').tolist()

plt.plot(range(len(GA_best)),GA_best)
plt.plot(range(len(DE_best)),DE_best)
plt.legend(['GA','DE'])
plt.ylabel('Predict value')
plt.xlabel('Iteration')
plt.show()
