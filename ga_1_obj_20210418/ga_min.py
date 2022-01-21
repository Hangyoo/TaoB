import random
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#初始化群体，群体的规模为5，每个染色体为（x1,x2,x3)的形式表示
#返回群体中的染色体矩阵
def init():
    x = (np.mat(np.random.rand(50, 7)))
    x *= 20
    x -= 10
    return x

#方程为f（x）= 1/（x1^2+x2^2+x3^2+x4^4+1)
#把每个群体中的染色体代入方程中，得到每个群体中染色体的适应值
#返回每个染色体的适应值
def adapt(x):
    f = []
    for args in x:
        # todo 在此修改目标函数
        # todo 当前目标函数为 100 / (|x1|+|x2|+|x3|+|x4|+|x5|+|x6|+1)
        val = 100 / (abs(args[0] + 1) + abs(args[1]) + abs(args[2] - 1) + abs(args[3] + 2) + abs(args[4] + 3) + abs(
            args[5] - 2) + abs(args[5] - 3) + 1)
        f.append(val)
    return f

#把适应值求和，求出每个染色体适应值与总和的比
#返回每个染色体适应值与总和的比
def select(f):
    sum = 0
    f1 = []
    for i in f:
        sum += i
    for i in f:
        i /= sum
        f1.append(i)
    return f1

#随机产生随机数，再与染色体适应值比，判断是否选中该染色体
#返回选中后的染色体
def select1(f, x):
    f1 = f.copy()
    c = np.random.rand(1,7).tolist()
    C = []
    f2 = []
    for i in c:
        sum = 0
        for k,j in enumerate(i):
            sum += f1[k]
            f1[k] = sum     #适应值的和
            C.append(j)
    for i in C:
        for j in range(len(f1)):
            if i < f1[j]:
                f2.append(f1.index(f1[j]))  #得到选中染色体的坐标
                break
    x1 = x.copy()
    for i,j in enumerate(f2):
        x1[i] = x[j]       #得到种群
    x1 = np.around(x1, decimals = 5)
    #print(x1)
    return x1

#交配率为0.85，随机产生每个染色体的随机数，判断是否参与交配
#在参与交配的染色体中再随机产生作为交配的交配位进行交配
#返回交配后的新群体
def copulation(f):
    f1 = []
    f2 = []
    c = np.random.rand(1, 50).tolist()
    for i in c:
        for j in i:
            if j < 0.85:     #交配概率
                f1.append(i.index(j))  #交配的染色体位置
    for i in f1:
        f2.append(f[i])           #交配的染色体
    for i in range(len(f1)):
        if i % 2 != 0:
            rand = random.randint(0,7)      #随机产生交配位
            for k in range(rand + 1, len(f2[0])):
                f2[i-1][k],f2[i][k] = f2[i][k],f2[i-1][k] #交配
    for i,j in enumerate(f1):
        f[j] = f2[i]
    return f

# 变异方式2: 模拟二进制交叉方式
def crossover_sbx(f):
    f1 = []
    f2 = []
    c = np.random.rand(1, 50).tolist()
    for i in c:
        for j in i:
            if j < 0.85:     #交配概率
                f1.append(i.index(j))  #交配的染色体位置
    for i in f1:
        f2.append(f[i])           #交配的染色体
    for i in range(len(f1)):
        if i % 2 != 0:
            rand = random.randint(0,7)      #随机产生交配位
            for k in range(rand + 1, len(f2[0])):
                #f2[i-1][count],f2[i][count] = f2[i][count],f2[i-1][count] #交配
                y1,y2 = f2[i][k],f2[i-1][k]
                # 模拟二进制SBX交叉操作,交叉参数根据经验设定为20
                r = random.random()
                eta_c = 20
                if r <= 0.5:
                    betaq = (2*r)**(1.0/(eta_c + 1.0))
                else:
                    betaq = (0.5/(1.0-r))**(1.0/(eta_c+1.0))
                child1 = 0.5 * ((1+betaq)*y1+(1-betaq)*y2)
                child2 = 0.5 * ((1-betaq)*y1+(1+betaq)*y2)
                # 解的修复
                child1 = min(20, max(child1,-20))
                child2 = min(20, max(child2,-20))
                f2[i][k],f2[i-1][k] = child1, child2

    for i,j in enumerate(f1):
        f[j] = f2[i]
    return f

# 随机选取一种交叉方式进行交叉
def crossover(f):
    if random.random() <= 0.5:
        return crossover_sbx(f)
    else:
        return copulation(f)

#变异的概率为0.1，每个染色体的每个基因都有可能会产生变异
#随机对应染色体的基因产生随机数，判断是否变异，
#如果变异，再在变异的基因上，随机产生一个基因接受范围的数
#返回变异后的新群体

# todo 机器学习技术引入变异中，在目标空间中对种群进行聚类，挑选每类中的一个代表解进行变异，使得算法在空间中的采样更均匀，算法的结果更优。
def variation_ml(f):
    # 采用Kmeans对解空间进行聚类，4类
    X = np.array(adapt(f)).reshape(50, 1)
    kmeans_model = KMeans(n_clusters=4, random_state=1).fit(X)
    labels = kmeans_model.labels_
    c = np.random.rand(50, 6) #染色体生成随机数
    c = np.where(c < 0.1, -1, c)  #判断随机数小于0.1为变异
    for n, i in enumerate(c):
        if (-1 in i):
            for m, j in enumerate(i):
                if j == -1:
                    # print('变异的位置：', n, m)
                    f[n][m] = np.random.rand() * 10 - 5  # 随机数替代变异数
        # 对第二类标签的个体进行变异
        elif (-1 in i) and labels[n] == 1:
            for m, j in enumerate(i):
                if j == -1:
                    # print('变异的位置：', n, m)
                    f[n][m] = np.random.rand() * 10 - 5  # 随机数替代变异数
        # 对第三类标签的个体进行变异
        elif (-1 in i) and labels[n] == 2:
            for m, j in enumerate(i):
                if j == -1:
                    # print('变异的位置：', n, m)
                    f[n][m] = np.random.rand() * 10 - 5  # 随机数替代变异数
        # 对第三类标签的个体进行变异
        elif (-1 in i) and labels[n] == 3:
            for m, j in enumerate(i):
                if j == -1:
                    # print('变异的位置：', n, m)
                    f[n][m] = np.random.rand() * 10 - 5  # 随机数替代变异数
        return f

def variation_nl(f):
    c = np.random.rand(50,7) #染色体生成随机数
    c = np.where(c < 0.1, -1, c)  #判断随机数小于0.1为变异
    for n, i in enumerate(c):
        if (-1 in i):
            for m, j in enumerate(i):
                if j == -1:
                    #print('变异的位置：', n, m)
                    f[n][m] = np.random.rand()  #随机数替代变异数
    return f

# 变异方式2: 多项式变异
def mutation_polynomial(f):
    c = np.random.rand(50, 7) #染色体生成随机数
    c = np.where(c < 0.1, -1, c)  #判断随机数小于0.1为变异
    for n, i in enumerate(c):
        if (-1 in i):
            for m, j in enumerate(i):
                if j == -1:
                    # 多项式变异
                    y = f[n][m]
                    delta1 = 1.0*(y+10)/20
                    delta2 = 1.0*(10-y)/20
                    eta_m = 0.5
                    mut_pow = 1.0 / (eta_m + 1)
                    r = random.random()
                    if r <= 0.5:
                        xy = 1 - delta1
                        val = 2.0 * r + (1.0 - 2.0*r) * (xy**(eta_m+1.0))
                        deltaq = val**mut_pow - 1.0
                    else:
                        xy = 1 - delta2
                        val = 2.0 * (1.0- r) + 2.0 * (r - 0.5) * (xy**(eta_m+1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y = y + deltaq*20
                    y = min(10, max(y,-10))
                    f[n][m] = y
    return f

def variation(f):
    if random.random() > 0.8:
        return variation_ml(f)
    else:
        if random.random() > 0.5:
            return variation_nl(f)
        else:
            return mutation_polynomial(f)



if __name__ == '__main__':
    x = init()            #返回群体的染色体
    x = x.tolist()        #转成列表形式
    f = adapt(x)          #返回每个染色体的适应值
    best = 100         #每个染色体的适应值的最优值
    gen = 10000
    best_list = [best]
    for i in range(gen):   #算法迭代一千次
        f1 = select(f)      #返回染色体适应值的比
        c = select1(f1, x)  #返回选择染色体后的种群
        C = copulation(c)   # 返回交配后的种群
        x = variation(C)    #返回变异后的种群
        f = adapt(x)        #重新评估适应值
        best1 = min(f)      #选择变异后的最优适应值
        if best > best1:    #判断每次变异后的最有适应值与选择出来的适应值，哪个更优，选择最优的那个
            best = best1
            # 找到最优解并记录
            best_idiv = x[f.index(best1)][:3]
        best_list.append(best)
        if i % 1000 == 0:
            print('第', i, '次循环：')
    print('循环100次的所有最小值：')
    print('最优解为：',best)

    plt.plot(range(gen+1),best_list)
    plt.xlabel('Iteration')
    plt.ylabel('Fuction value')
    plt.show()