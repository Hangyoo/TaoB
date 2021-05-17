import random
from operator import itemgetter
from matplotlib import pyplot as plt

CXPB = 0.8  # 交叉概率
MUTPB = 0.08  # 变异概率
NGEN = 10000  # 迭代次数
popsize = 20  # 种群个体数

# ==============
d = 4 # 决策变量的个数
# ===============
lb = [-5.0 for _ in range(d)] # 每个决策变量的下界设置为-5
ub = [5.0 for _ in range(d)]  # 每个决策变量的上界设置为5

# 染色体基因
class Gene():
    '''
    gene: n维的数组
    '''

    def __init__(self, gene):
        self.gene = gene # 将局部变量gen赋值给类属性gen

# 遗传算法框架
class GA():
    '''
    初始化种群以及保存种群的最好个体
    '''

    def __init__(self):
        pop = [] # 种群
        for i in range(popsize):  # popsize 种群大小
            individual = [] # 个体
            for j in range(len(ub)): # 根据每个维度的上下界生成个体
                individual.append(random.uniform(lb[j], ub[j]))
            fitness = self.evaluate(individual) # 评估个体的适应度值
            pop.append({'Gene': Gene(individual), 'fitness': fitness}) # 将个体编码和适应度值加入种群中
        self.pop = pop # 将局部变量pop赋值给类属性pop
        self.bestIndividual = self.selectBest(pop) # 选择种群中最好的个体

    '''
    评估一个个体的好坏，找最大值
    y:适应度，f1 = 1 / (x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + 1)
    '''

    def evaluate(self, x):
        y = 1 / (x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + 1)
        return y  # 适应度必须越大越好

    '''
    选择种群中最好的个体
    Pop：种群
    '''
    def selectBest(self, pop):
        s = sorted(pop, key=itemgetter('fitness'), reverse=True)  # 按照适应度从大到小排列
        return s[0]

    '''
    选择
    pop:种群
    '''
    def selction(self, pop):
        # 轮盘赌选择方式
        sum_fit = sum(ind['fitness'] for ind in pop)  # 计算适应度值总和
        chosen = [] # 选择的个体将会加入chosen列表中
        for i in range(popsize): # 对种群进行遍历，选择解(具体参考轮盘赌选择)
            fit = sum_fit * random.random()
            t = 0
            for ind in pop:
                t += ind['fitness']
                if t >= fit:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=itemgetter('fitness'), reverse=False)  # 从小到大排序，因为pop方法会弹最后一个出来
        return chosen

    '''
    交叉：双点交叉（了解就行，交叉方式很多种，每种的编码方式都不一致）
    offspring：两个父代个体
    功能：两个父代个体执行两点交叉产生两个子代个体
    '''
    def cross(self, offspring):
        dim = len(offspring[0]['Gene'].gene)  # 获取染色体编码长度
        gen1 = offspring[0]['Gene'].gene      # 获取父代1的染色体
        gen2 = offspring[1]['Gene'].gene      # 获取父代2的染色体

        if dim == 0:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)   # 获取交叉点1
            pos2 = random.randrange(1, dim)   # 获取交叉点2

        newOff1 = Gene([])  # 子代个体1
        newOff2 = Gene([])  # 子代个体2
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp1.append(gen1[i])
                temp2.append(gen2[i])
            else:
                temp1.append(gen2[i])
                temp2.append(gen1[i])
        newOff1.gene = temp1
        newOff2.gene = temp2
        f1 = self.evaluate(newOff1.gene) # 评估子代个体1
        f2 = self.evaluate(newOff2.gene) # 评估子代个体2
        return {'Gene': newOff1, 'fitness': f1}, {'Gene': newOff2, 'fitness': f2}

    '''
    变异
    individual：父代个体
    功能：单个父代个体执行单点变异产生单个子代个体
    '''
    def mut(self, individual):
        dim = len(individual['Gene'].gene) # 获取个体染色体长度
        if dim == 1: # 随机选择一个位置pos进行变异
            pos = 0
        else:
            pos = random.randrange(0, dim)
        individual['Gene'].gene[pos] = random.uniform(lb[pos], ub[pos]) # 在上界和下界中随机选择一个数替换原pos位置的基因
        individual['fitness'] = self.evaluate(individual['Gene'].gene)  # 评估新的染色体
        return individual

    '''遗传算法的主程序'''
    def GA_main(self):
        all_y = []
        for g in range(NGEN): # 迭代次数
            selectPop = self.selction(self.pop)
            nextPop = []
            while len(nextPop) != popsize:
                offspring = [selectPop.pop() for _ in range(2)]  # 取两个个体用于交叉和变异操作
                if random.random() <= CXPB:  # 交叉
                    cf1, cf2 = self.cross(offspring)
                    if random.random() <= MUTPB:
                        mu1 = self.mut(cf1)
                        mu2 = self.mut(cf2)
                        nextPop.append(mu1)
                        nextPop.append(mu2)
                    else:
                        nextPop.append(cf1)
                        nextPop.append(cf2)
                else:  #
                    nextPop.extend(offspring)

            self.pop = nextPop #将新种群复制给self.pop，保证算法可以不断迭代

            # 轮盘赌选择操作
            bestIndividual = self.selectBest(self.pop)

            # 保存最好值
            if bestIndividual['fitness'] > self.bestIndividual['fitness']:
                self.bestIndividual = bestIndividual

            if g == NGEN - 1:
                # 结果保存
                print("函数最大值：{}".format(self.bestIndividual['fitness']))
                print("在x={}时取得".format(self.bestIndividual['Gene'].gene))

                # print("bestindividual x: {}, fit:{}".format(self.bestIndividual['Gene'].gene, 100.0/self.bestIndividual['fitness']))
            all_y.append(self.bestIndividual['fitness'])

        # 绘图
        plt.xlabel('iteration:')
        plt.ylabel('y:')

        plt.plot(all_y)
        plt.show()


if __name__ == "__main__":
    ga = GA()
    ga.GA_main()