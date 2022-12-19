import random, math, sys
import matplotlib.pyplot as plt # 画图
from copy import deepcopy
import copy
from tqdm import *  # 进度条
from util import *
import numpy as np
import pandas as pd

DEBUG = False

'''
问题：求解带时间窗的车辆路径调度问题
求解算法：NSGA-II
目标函数：1.路径最短 2.成本最低    
'''
sampleSolution = [0, 1, 20, 9, 3, 12, 26, 25, 24, 4, 23, 22, 21, 0, 13, 2, 15, 14, 6, 27, 5, 16, 17, 8, 18, 0, 7, 19,
                  11, 10, 0]

geneNum = 200  # 种群数量
generationNum = 200  # 迭代次数

CENTER = 0  # 配送中心

HUGE = 9999999
VARY = 0.1  # 变异几率

n = 14  # 客户点数量
m = 0
k = 2  # 车辆数量
Q = 2  # 额定载重量, t
dis = 1e10  # 续航里程, km
costPerKilo = 1  # 油价
epu = 1  # 早到惩罚成本
lpu = 1  # 晚到惩罚成本
speed = 50  # 速度，km/h

# 坐标
X = [113.312146, 113.336186, 113.580131, 113.275097, 113.345022, 113.340205, 113.356905, 113.315112,\
     113.347743, 113.370387, 113.369274, 113.302336, 113.26229, 113.293756, 113.299985]
Y = [23.402795, 23.194567, 23.556148, 23.087826, 23.108368, 23.100978, 23.131771, 23.162686, \
     23.139855, 23.124197, 23.140314, 23.133698, 23.115563, 23.131901, 23.130197]

# 需求量
t = [0, 0.418, 0.159, 0.258, 0.027, 0.234, 0.211, 0.021, 0.309, 0.171, 0.121, 0.111, 0.318, 0.536, 0.415, \
     0.4, 0.1, 0.1, 0.2, 0.5, 0.2, 0.7, 0.2, 0.7, 0.1, 0.5, 0.4, 0.4]

# 最早到达时间
# eh = [0, 0, 1, 2, 7, 5, 3, 0, 7, 1, 4, 1, 3, 0, 2, 2, 7, 6, 7, 1, 1, 8, 6, 7, 6, 4, 0, 0]
eh = [0]*28

# 最晚到达时间
# lh = [100, 1, 2, 4, 8, 6, 5, 2, 8, 3, 5, 2, 4, 1, 4, 3, 8, 8, 9, 3, 3, 10, 10, 8, 7, 6, 100, 100]
lh = [1e4]*28

# 服务时间
# h = [0, 0.2, 0.3, 0.3, 0.3, 0.3, 0.5, 0.8, 0.4, 0.5, 0.7, 0.7, 0.6, 0.2, 0.2, 0.4, 0.1, 0.1, 0.2, 0.5, 0.2, 0.7, 0.2,
     # 0.7, 0.1, 0.5, 0.4, 0.4]
h = [0]*28

Ji = [0.0810,0.3654,0.0688,0.0188,0.0507,0.0470,0.0252,0.0535,0.0433,0.0445,0.0205,0.0659,0.0644,0.0510]

jam = np.array(pd.read_excel('./jam.xls'))
distance = np.array(pd.read_excel('./distance.xls'))

class Gene:
    def __init__(self, name='Gene', data=None):
        self.name = name
        self.length = n + m + 1
        if data is None:
            self.data = self._getGene(self.length)
        else:
            # assert(self.length+k == len(data))
            self.data = data
        self.fit = self.getFit()
        self.chooseProb = 0  # 选择概率

    # randomly choose begin gene
    def _generate(self, length):
        data = [i for i in range(1, length)]
        random.shuffle(data)
        data.insert(0, CENTER)
        data.append(CENTER)
        return data

    # insert zeors at proper positions
    def _insertZeros(self, data):
        sum = 0
        newData = []
        for index, pos in enumerate(data):
            sum += t[pos]
            if sum > Q:
                newData.append(CENTER)
                sum = t[pos]
            newData.append(pos)
        return newData

    # return begin random gene with proper center assigned
    def _getGene(self, length):
        data = self._generate(length)
        data = self._insertZeros(data)
        return data

    # return fitness
    def getFit(self):
        fit = distCost = timeCost = overloadCost = fuelCost = 0
        dist = []  # from this to next

        # calculate distance
        i = 1
        while i < len(self.data):
            calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
            dist.append(calculateDist(X[self.data[i]], Y[self.data[i]], X[self.data[i - 1]], Y[self.data[i - 1]]))
            i += 1

        # distance cost
        distCost = sum(dist) * costPerKilo

        # time cost
        timeSpent = 0
        for i, pos in enumerate(self.data):
            # skip first center
            if i == 0:
                continue
            # new car
            elif pos == CENTER:
                timeSpent = 0
            # update time spent on road
            timeSpent += (dist[i - 1] / speed)
            # arrive early
            if timeSpent < eh[pos]:
                timeCost += ((eh[pos] - timeSpent) * epu)
                timeSpent = eh[pos]
            # arrive late
            elif timeSpent > lh[pos]:
                timeCost += ((timeSpent - lh[pos]) * lpu)
            # update time
            timeSpent += h[pos]

        # overload cost and out of fuel cost
        load = 0
        distAfterCharge = 0
        for i, pos in enumerate(self.data):
            # skip first center
            if i == 0:
                continue
            # charge here
            if pos > n:
                distAfterCharge = 0
            # at center, re-load
            elif pos == CENTER:
                load = 0
                distAfterCharge = 0
            # normal
            else:
                load += t[pos]
                distAfterCharge += dist[i - 1]
                # update load and out of fuel cost
                overloadCost += (HUGE * (load > Q))
                fuelCost += (HUGE * (distAfterCharge > dis))

        fit = distCost + timeCost + overloadCost + fuelCost
        #print(sum(dist),fit)
        return 1/fit

    def updateChooseProb(self, sumFit):
        self.chooseProb = self.fit / sumFit

    def moveRandSubPathLeft(self):
        # pass
        path = random.randrange(k)  # choose begin path index
        index = self.data.index(CENTER, path+1) # move to the chosen index
        # move first CENTER
        locToInsert = 0
        self.data.insert(locToInsert, self.data.pop(index))
        index += 1
        locToInsert += 1
        # move data after CENTER
        while self.data[index] != CENTER:
            self.data.insert(locToInsert, self.data.pop(index))
            index += 1
            locToInsert += 1

        # assert(self.length+k == len(self.data))

    # plot this gene in begin new window
    def plot(self):
        Xorder = [X[i] for i in self.data]
        Yorder = [Y[i] for i in self.data]
        plt.plot(Xorder, Yorder, c='black', zorder=1)
        plt.scatter(X, Y, zorder=2)
        plt.scatter([X[0]], [Y[0]], marker='o', zorder=3)
        # plt.scatter(X[-m:], Y[-m:], marker='^', zorder=3)
        plt.title(self.name)
        plt.show()

def cal_fit(data):
    timeCost = overloadCost = fuelCost = 0
    dist = []  # from this to next

    # calculate distance
    i = 1
    while i < len(data):
        calculateDist = lambda x1, y1, x2, y2: math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
        dist.append(calculateDist(X[data[i]], Y[data[i]], X[data[i - 1]], Y[data[i - 1]]))
        i += 1

    # 计算目标函数1
    fun1 = [0]
    # 计算目标f2
    fun2 = []
    new_data = []
    for i in range(len(data)):
        if data[i] == 0 and (i not in [0,len(data)-1]):
            pass
        else:
            new_data.append(data[i])

    new_dist = []
    for i in range(len(new_data)-1):
        new_dist.append(distance[new_data[i],new_data[i+1]])
        fun2.append(jam[new_data[i],new_data[i+1]])
    for k in range(len(Ji)):
        fun1.append(Ji[k]*new_dist[k]/speed)




    # distance cost
    distCost = sum(dist) * costPerKilo

    # time cost
    timeSpent = 0
    for i, pos in enumerate(data):
        # skip first center
        if i == 0:
            continue
        # new car
        elif pos == CENTER:
            timeSpent = 0
        # update time spent on road
        timeSpent += (dist[i - 1] / speed)
        # arrive early
        if timeSpent < eh[pos]:
            timeCost += ((eh[pos] - timeSpent) * epu)
            timeSpent = eh[pos]
        # arrive late
        elif timeSpent > lh[pos]:
            timeCost += ((timeSpent - lh[pos]) * lpu)
        # update time
        timeSpent += h[pos]

    # overload cost and out of fuel cost
    load = 0
    distAfterCharge = 0
    for i, pos in enumerate(data):
        # skip first center
        if i == 0:
            continue
        # charge here
        if pos > n:
            distAfterCharge = 0
        # at center, re-load
        elif pos == CENTER:
            load = 0
            distAfterCharge = 0
        # normal
        else:
            load += t[pos]
            distAfterCharge += dist[i - 1]
            # update load and out of fuel cost
            overloadCost += (HUGE * (load > Q))
            fuelCost += (HUGE * (distAfterCharge > dis))

    fit = distCost + timeCost + overloadCost + fuelCost
    return sum(fun1), sum(fun2)


def getSumFit(genes):
    sum = 0
    for gene in genes:
        sum += gene.fit
    return sum


# return begin bunch of random genes
def getRandomGenes(size):
    genes = []
    for i in range(size):
        genes.append(Gene("Gene "+str(i)))
    return genes


# 计算适应度和
def getSumFit(genes):
    sumFit = 0
    for gene in genes:
        sumFit += gene.fit
    return sumFit


# 更新选择概率
def updateChooseProb(genes):
    sumFit = getSumFit(genes)
    for gene in genes:
        gene.updateChooseProb(sumFit)


# 计算累计概率
def getSumProb(genes):
    sum = 0
    for gene in genes:
        sum += gene.chooseProb
    return sum


# 选择复制，选择前 1/3
def choose(genes):
    num = int(geneNum/6) * 2    # 选择偶数个，方便下一步交叉
    # sort genes with respect to chooseProb
    key = lambda gene: gene.chooseProb
    genes.sort(reverse=True, key=key)
    # return shuffled top 1/3
    return genes[0:num]


# 交叉一对
def crossPair(gene1, gene2, crossedGenes):
    gene1.moveRandSubPathLeft()
    gene2.moveRandSubPathLeft()
    newGene1 = []
    newGene2 = []
    # copy first paths
    centers = 0
    firstPos1 = 1
    for pos in gene1.data:
        firstPos1 += 1
        centers += (pos == CENTER)
        newGene1.append(pos)
        if centers >= 2:
            break
    centers = 0
    firstPos2 = 1
    for pos in gene2.data:
        firstPos2 += 1
        centers += (pos == CENTER)
        newGene2.append(pos)
        if centers >= 2:
            break
    # copy data not exits in father gene
    for pos in gene2.data:
        if pos not in newGene1:
            newGene1.append(pos)
    for pos in gene1.data:
        if pos not in newGene2:
            newGene2.append(pos)
    # add center at end
    newGene1.append(CENTER)
    newGene2.append(CENTER)
    # 计算适应度最高的
    key = lambda gene: gene.fit
    possible = []
    while gene1.data[firstPos1] != CENTER:
        newGene = newGene1.copy()
        newGene.insert(firstPos1, CENTER)
        newGene = Gene(data=newGene.copy())
        possible.append(newGene)
        firstPos1 += 1
    possible.sort(reverse=True, key=key)

    # crossedGenes.append(possible[0])
    # key = lambda gene: gene.fit
    # possible = []
    # while gene2.data[firstPos2] != CENTER:
    #     newGene = newGene2.copy()
    #     newGene.insert(firstPos2, CENTER)
    #     newGene = Gene(data=newGene.copy())
    #     possible.append(newGene)
    #     firstPos2 += 1
    # possible.sort(reverse=True, key=key)
    if possible == []:
        crossedGenes.append(gene1)
    else:
        crossedGenes.append(possible[0])


# 交叉
def cross(genes):
    crossedGenes = []
    for i in range(0, len(genes), 2):
        crossPair(genes[i], genes[i+1], crossedGenes)
    return crossedGenes


# 合并
def mergeGenes(genes, crossedGenes):
    # sort genes with respect to chooseProb
    key = lambda gene: gene.chooseProb
    genes.sort(reverse=True, key=key)
    pos = geneNum - 1
    for gene in crossedGenes:
        genes[pos] = gene
        pos -= 1
    return  genes


# 变异一个
def varyOne(gene):
    varyNum = 10
    variedGenes = []
    for i in range(varyNum):
        p1, p2 = random.choices(list(range(1,len(gene.data)-2)), k=2)
        newGene = gene.data.copy()
        newGene[p1], newGene[p2] = newGene[p2], newGene[p1] # 交换
        variedGenes.append(Gene(data=newGene.copy()))
    key = lambda gene: gene.fit
    variedGenes.sort(reverse=True, key=key)
    return variedGenes[0]


# 变异
def vary(genes):
    for index, gene in enumerate(genes):
        # 精英主义，保留前三十
        if index < 30:
            continue
        if random.random() < VARY:
            genes[index] = varyOne(gene)
    return genes


if __name__ == "__main__" and not DEBUG:

    population = getRandomGenes(geneNum)  # 初始种群
    genes = copy.deepcopy(population)
    # 迭代
    gen = 0
    for i in tqdm(range(generationNum)):
        updateChooseProb(genes)
        sumProb = getSumProb(genes)
        chosenGenes = choose(deepcopy(genes))  # 选择
        crossedGenes = cross(chosenGenes)  # 交叉
        offspring1 = mergeGenes(genes, crossedGenes)  # 复制交叉至子代种群
        offspring2 = vary(genes)  # 变异

        Chrom = population + offspring1 + offspring2 # 合并种群

        chroms_obj_record = {}

        for i in range(geneNum * 3):  # 计算每个体的目标函数值 ={chromosome:[距离,费用]}
            data = Chrom[i].data
            dis, cost = cal_fit(data)
            chroms_obj_record[i] = [dis, cost]

        front = Non_donminated_sorting(geneNum, chroms_obj_record)
        population, new_popindex = Selection(geneNum, front, chroms_obj_record, Chrom)
        # 最优结果保留
        best_list, best_obj = [], []
        if gen == generationNum - 1:
            for j in front[0]:
                best_list.append(Chrom[j])
                best_obj.append(chroms_obj_record[j])
        gen += 1
    print(f'NSGAII求解到的非支配解个数为:{len(best_list)}个')
    print('染色体编码:', best_list[0].data)
    f1, f2 = cal_fit(best_list[0].data)
    print(f'f1:{f1},f2：{f2}')
    best_list[0].plot() # 画出来

    ##### 绘制Pareto图
    X = []
    Y = []
    for item in best_obj:
        dis, cost = item
        X.append(dis)
        Y.append(cost)
    plt.plot(X,Y,'o')
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title('Pareto Front')
    plt.show()