__author__ = 'lly'

##拉丁超立方采样函数，均匀的产生随机参数样本，n_sample=popSize
def latin_hypercube_sampling(n_param=8, n_sample=100):
    index = np.arange(n_sample)
    np.random.shuffle(index)

    for i in range(n_param-1):
        r = np.arange(n_sample)
        np.random.shuffle(r)
        index = np.vstack((index, r))

    params = np.zeros((n_param, n_sample))
    for i in range(n_param):
        params[i] = (param_range[i][1] - param_range[i][0]) * (index[i] + np.random.random(n_sample)) / n_sample + param_range[i][0]

    return params.T##params.T是n_sample行决策变量组合数组

##定义一个个体，一个个体是一组决策变量（b,a1,a2)
class Individual(object):
    def __init__(self, n_param): # 初始化个体属性
        self.n_param = n_param##获得个体参数
        self.params = np.zeros(self.n_param)

    def assignFeatures(self, arr): ###赋予个体新的参数值
        self.params = arr

    def calculateObj1(self): #计算目标函数1值
        self.objective1 = function1(self.params[0],
                                    self.params[1],
                                    self.params[2],
                                    self.params[3],
                                    self.params[4],
                                    self.params[5],
                                    self.params[6],
                                    self.params[7])
        return self.objective1

    def calculateObj2(self,): #计算目标函数2值
        self.objective2 = function2(self.params[0],
                                    self.params[1],
                                    self.params[2],
                                    self.params[3],
                                    self.params[4],
                                    self.params[5],
                                    self.params[6],
                                    self.params[7])
        return self.objective2

    def __del__(self): #删除一个个体
        self.objective1 = None
        self.objective2 = None
        self.params = None

##定义一个种群
class Population(object):
    def __init__(self):
        self.population = []
    def __len__(self): # 计算种群数量
        return len(self.population)
    def __iter__(self): # 允许种群迭代个体？？
        return self.population.__iter__()
    def __getitem__(self, key):##得到种群中的一个个体
        return self.population[key]
    def __setitem__(self, key, individual):##为种群中的一个个体赋值
        self.population[key] = individual
    def addIndividual(self, newIndividual): #向种群中加入新个体
        self.population.append(newIndividual)
    def __del__(self): #删除种群
        self.population = None

##初始化一个种群函数
def createInitialPopulation(n_param,popSize):
    random_params = latin_hypercube_sampling(n_param, popSize)##用拉丁超立方采样函数产生popSize组决策变量
    population = Population()##生成一个空种群

    for i in range(popSize):
        individual = Individual(n_param)##生成一个个体
        individual.assignFeatures(random_params[i,:])##将随机生成的决策变量赋予这个个体
        population.addIndividual(individual)##个体加入到种群中，从而形成一个初始种群
    return population

##得到目标值集合函数
def objectiveValueSet(population,popSize):##对种群种的每个个体分别计算目标函数值并生成列表
    value1List=[0 for i in range(popSize)]
    value2List=[0 for i in range(popSize)]

    for i in range(popSize):
        value1List[i]=round(population[i].calculateObj1(),2)
        value2List[i]=round(population[i].calculateObj2(),2)

    return value1List,value2List,


##快速非支配函数
def fast_non_dominated_sort(population,popSize,n_param):
    ##构建列表并把目标函数计算结果存放在列表里
    function1List=objectiveValueSet(population,popSize)[0]
    function2List=objectiveValueSet(population,popSize)[1]

    #define the dominate set Sp
    dominateList=[set() for i in range(popSize)]#populationSize个set()元素的列表
    #define the dominated set
    dominatedList=[set() for i in range(popSize)]
    #compute the dominate and dominated entity for every entity in the population
    for p in range(popSize):
        for q in range(popSize):
            ##p支配q
            if function1List[p]< function1List[q] and function2List[p]<function2List[q]:
                dominateList[p].add(q)
            elif function1List[p]> function1List[q] and function2List[p]>function2List[q]:
                dominatedList[p].add(q)
     #compute dominated degree Np
    for i in range(len(dominatedList)):
        dominatedList[i]=len(dominatedList[i])
    #create list to save the non-dominated front information
    NDFSet=[]
    #compute non-dominated front
    while max(dominatedList)>=0:
        front=[]
        for i in range(len(dominatedList)):
            if dominatedList[i]==0:
                front.append(i)
        NDFSet.append(front)
        for i in range(len(dominatedList)):
            dominatedList[i]=dominatedList[i]-1
    return NDFSet##NDFSet是元素为不同支配层个体序号的列表

#计算拥挤距离的函数
def crowdedDistance(population,popSize,Front):##Front是指某层的个体的序号集合
    distance=pd.Series([float(0) for i in range(len(Front))], index=Front)#初始化个体间的拥挤距离，生成一个有len(front)个行0的data frame
    ##利用这层个体序号从population中将对应的个体（一组决策变量）调出来
    FrontSet=[]
    for i in Front:
        FrontSet.append(population[i])
    ##保存这层个体的目标函数值
    function1_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[0]
    function2_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[1]

    function1Ser=pd.Series(function1_Front_List,index=Front)
    function2Ser=pd.Series(function2_Front_List,index=Front)

    ##目标函数值排序
    function1Ser.sort_values(ascending=False,inplace=True)
    function2Ser.sort_values(ascending=False,inplace=True)

    print('function test')
    print(function1Ser)
    print(function2Ser)

##设置这层中目标函数值最大和最小个体之间的距离
    distance[function1Ser.index[0]]=1000
    distance[function1Ser.index[-1]]=1000
    distance[function2Ser.index[0]]=1000
    distance[function2Ser.index[-1]]=1000

    ##计算其他个体的distance值
    for i in range(1,len(Front)-1):
        distance[function1Ser.index[i]]=distance[function1Ser.index[i+1]]+(function1Ser[function1Ser.index[i-1]]-function1Ser[function1Ser.index[i-1]])/(max(function1_Front_List)-min(function1_Front_List))
        distance[function2Ser.index[i]]+=(function2Ser[function2Ser.index[i+1]]-function2Ser[function2Ser.index[i-1]])/(max(function2_Front_List)-min(function2_Front_List))

    distance.sort_values(ascending=False,inplace=True)
    print('distance is')
    print(distance)
    return distance##dataframe byte=float64


##交叉函数
def crossover(ind1,ind2,n_param): #模拟二进制交叉算子产生子代
    random.seed()
    child1 = Individual(n_param)
    child2 = Individual(n_param)
    geneIndex = list(range(n_param))

    crossoverWeight = (ind1.calculateObj1()+1)/ (ind1.calculateObj1() + ind2.calculateObj1() + 2)
    crossoverWeight = 0.5 if crossoverWeight < 0.01 else crossoverWeight

    if random.random() <= 0.9: # crossover probability
            halfGeneIndex = random.choice(geneIndex)
            for gene in geneIndex:
                if gene==halfGeneIndex: # SBX crossover
                    child1.params[gene] = crossoverWeight* ind1.params[gene] + 0.5 * ind2.params[gene]
                    child2.params[gene] = ind2.params[gene]
                else:
                    child1.params[gene] = ind1.params[gene]
                    child2.params[gene] = crossoverWeight * ind1.params[gene] + 0.5* ind2.params[gene]

    else: # no crossover, i.e., maintain all the parent features
            child1.params = ind1.params.copy()
            child2.params = ind2.params.copy()

    return child1, child2

##变异函数
def gaussMutationStrength(generation): #随机产生变异长度
        random.seed()
        #return rd.normal(0, 0.2 * np.exp(-(1 - generation / generationMax)))
        return np.random.normal(0, 0.05 * (1 - generation / generationMax))

def mutate(child, generation): # 产生子代的变异函数
    random.seed()
    for i in range(8):
        if random.random() <= 0.1: # mutation probability
            child.params[i] = child.params[i] + gaussMutationStrength(generation) * (param_range[i][1] - param_range[i][0])

            child.params[i] = param_range[i][1] if child.params[i] > param_range[i][1] else child.params[i]
            child.params[i] = param_range[i][0] if child.params[i] < param_range[i][0] else child.params[i]

    return child

#查找列表指定元素的索引,查找a在列表list里的索引
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

##将values列表中的个体从小到大排列并用个体索引表示
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        # 当结果长度不等于初始长度时，继续循环
        if index_of(min(values),values) in list1:##如果values这个列表中最小值的索引在list1中
            sorted_list.append(index_of(min(values),values))##在sorted_list中增加该索引
        values[index_of(min(values),values)] = math.inf#删除values列表中的最小值
    print(sorted_list)
    return sorted_list



##设置种群参数
n_param=8
popSize=10
#创造一个最初的种群
population = createInitialPopulation(n_param,popSize)
print ('Complete initializing the population...')
##while循环,产生帕累托最优解
time_start=time.time() #记录循环开始时间
generationMax=1
generation = 1
while generation <= generationMax:##当达到最大代时，或满足收敛标准且没有现有的支配解时，退出优化迭代
    print ('Start optimization at generation %d...' % generation)
    function1_value_list=objectiveValueSet(population,popSize)[0]
    function2_value_list=objectiveValueSet(population,popSize)[1]

    non_dominated_sorted_solution=fast_non_dominated_sort(population,popSize,n_param)##对population进行非支配层分层
    first_front=non_dominated_sorted_solution[0]##储存这代种群个体最优非支配层个体序号
    population2=population##population2是拥有2倍popSize的种群，先将他的一半个体赋值为population
    for j in range(0, popSize, 2):
        child1, child2 = crossover(population[j], population[j+1],n_param)##将population相邻的两个个体交叉形成两个新的个体child1,child2
        mutate(child1, generation)##将child1和child2再进行变异形成新的个体
        mutate(child2, generation)
        population2.addIndividual(child1)
        population2.addIndividual(child2)##在population2中加入交叉变异形成的新的个体最后形成父代和子代混合的种群population2
    non_dominated_sorted_solution2=fast_non_dominated_sort(population2,popSize*2,n_param)##对population2进行非支配层分层
    print('loop test')
    print('non_dominated_sorted_solution2')
    print(non_dominated_sorted_solution2)
    crowd_distance_front=[]##储存这层个体的拥挤距离列表
    crowding_distance_values2=[]#元素为每层拥挤距离列表的列表
    for i in range(0,len(non_dominated_sorted_solution2)):##遍历非支配层中的每一层
        ##distance为该支配层个体的拥挤距离，因为是data frame(byte=float64)，要将其转换成列表
        print(i)
        distance=crowdedDistance(population2,popSize*2,non_dominated_sorted_solution2[i])#.astype(pd.np.int64)##转换float64
        print(distance)
        distance=distance.values##获得data frame的值
        print(distance)
        for j in distance:
            crowd_distance_front.append(j)
        crowding_distance_values2.append(crowd_distance_front)
        print('crowding_distance_values2')
        print(crowding_distance_values2)
    new_population_order= []##储存从population2中选出的一半优秀个体的索引
    for i in range(0,len(non_dominated_sorted_solution2)):##遍历非支配层中的每一层
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]##[0,1,2...]有该层个体个元素
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])#对该层个体的拥挤距离排序并用个体索引表示个体
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]#将个体索引换成在population2种群中的索引
        front.reverse()##将列表反向排列，使拥挤距离排序方式变成由大到小
        for j in front:##遍历front里的序号
            #将population2排序后前面一半优秀个体的序号放到列表new_population_order.append(j)中
            #排序方式是先从面的非支配层选取，当选到临界非支配层，即个体数超出popSize,将这层个体从拥挤距离大的选起
            new_population_order.append(j)
            if(len(new_population_order)==popSize):
                break
        if (len(new_population_order) == popSize):
            break
    print('new_population_order')
    print(new_population_order)
    population =createInitialPopulation(n_param,popSize=100)##重新生成一个种群
    for i,j in zip(new_population_order,range(0,popSize)):
        population.__setitem__(j,population2[i])##将这个种群的个体赋值为population2中的优秀个体
    generation+=1
    print('generation is',generation)
print ('total cpu time: %7.4f s' % (time.time() - time_start))


#保存结果
columns=['a1','a2', 'a3', 'a4','b1','b2', 'c1', 'c2',]
outputResult = pd.DataFrame(np.zeros((len(first_front), n_param+1)), columns=columns)
for i in range(len(first_front)):
    outputResult.iloc[i, 1:] = population[first_front[i]].params

outputResult.to_excel('optimization_result.xlsx')

##画图
Y= [i for i in function1_value_list]##环境效益
X= [k for k in function2_value_list]##成本
plt.xlabel('cost', fontsize=15)
plt.ylabel('environment', fontsize=15)
plt.scatter(X, Y)
plt.show()

