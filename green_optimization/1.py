__author__ = 'lly'

##�����������������������ȵĲ����������������n_sample=popSize
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

    return params.T##params.T��n_sample�о��߱����������

##����һ�����壬һ��������һ����߱�����b,a1,a2)
class Individual(object):
    def __init__(self, n_param): # ��ʼ����������
        self.n_param = n_param##��ø������
        self.params = np.zeros(self.n_param)

    def assignFeatures(self, arr): ###��������µĲ���ֵ
        self.params = arr

    def calculateObj1(self): #����Ŀ�꺯��1ֵ
        self.objective1 = function1(self.params[0],
                                    self.params[1],
                                    self.params[2],
                                    self.params[3],
                                    self.params[4],
                                    self.params[5],
                                    self.params[6],
                                    self.params[7])
        return self.objective1

    def calculateObj2(self,): #����Ŀ�꺯��2ֵ
        self.objective2 = function2(self.params[0],
                                    self.params[1],
                                    self.params[2],
                                    self.params[3],
                                    self.params[4],
                                    self.params[5],
                                    self.params[6],
                                    self.params[7])
        return self.objective2

    def __del__(self): #ɾ��һ������
        self.objective1 = None
        self.objective2 = None
        self.params = None

##����һ����Ⱥ
class Population(object):
    def __init__(self):
        self.population = []
    def __len__(self): # ������Ⱥ����
        return len(self.population)
    def __iter__(self): # ������Ⱥ�������壿��
        return self.population.__iter__()
    def __getitem__(self, key):##�õ���Ⱥ�е�һ������
        return self.population[key]
    def __setitem__(self, key, individual):##Ϊ��Ⱥ�е�һ�����帳ֵ
        self.population[key] = individual
    def addIndividual(self, newIndividual): #����Ⱥ�м����¸���
        self.population.append(newIndividual)
    def __del__(self): #ɾ����Ⱥ
        self.population = None

##��ʼ��һ����Ⱥ����
def createInitialPopulation(n_param,popSize):
    random_params = latin_hypercube_sampling(n_param, popSize)##������������������������popSize����߱���
    population = Population()##����һ������Ⱥ

    for i in range(popSize):
        individual = Individual(n_param)##����һ������
        individual.assignFeatures(random_params[i,:])##��������ɵľ��߱��������������
        population.addIndividual(individual)##������뵽��Ⱥ�У��Ӷ��γ�һ����ʼ��Ⱥ
    return population

##�õ�Ŀ��ֵ���Ϻ���
def objectiveValueSet(population,popSize):##����Ⱥ�ֵ�ÿ������ֱ����Ŀ�꺯��ֵ�������б�
    value1List=[0 for i in range(popSize)]
    value2List=[0 for i in range(popSize)]

    for i in range(popSize):
        value1List[i]=round(population[i].calculateObj1(),2)
        value2List[i]=round(population[i].calculateObj2(),2)

    return value1List,value2List,


##���ٷ�֧�亯��
def fast_non_dominated_sort(population,popSize,n_param):
    ##�����б���Ŀ�꺯��������������б���
    function1List=objectiveValueSet(population,popSize)[0]
    function2List=objectiveValueSet(population,popSize)[1]

    #define the dominate set Sp
    dominateList=[set() for i in range(popSize)]#populationSize��set()Ԫ�ص��б�
    #define the dominated set
    dominatedList=[set() for i in range(popSize)]
    #compute the dominate and dominated entity for every entity in the population
    for p in range(popSize):
        for q in range(popSize):
            ##p֧��q
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
    return NDFSet##NDFSet��Ԫ��Ϊ��֧ͬ��������ŵ��б�

#����ӵ������ĺ���
def crowdedDistance(population,popSize,Front):##Front��ָĳ��ĸ������ż���
    distance=pd.Series([float(0) for i in range(len(Front))], index=Front)#��ʼ��������ӵ�����룬����һ����len(front)����0��data frame
    ##������������Ŵ�population�н���Ӧ�ĸ��壨һ����߱�����������
    FrontSet=[]
    for i in Front:
        FrontSet.append(population[i])
    ##�����������Ŀ�꺯��ֵ
    function1_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[0]
    function2_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[1]

    function1Ser=pd.Series(function1_Front_List,index=Front)
    function2Ser=pd.Series(function2_Front_List,index=Front)

    ##Ŀ�꺯��ֵ����
    function1Ser.sort_values(ascending=False,inplace=True)
    function2Ser.sort_values(ascending=False,inplace=True)

    print('function test')
    print(function1Ser)
    print(function2Ser)

##���������Ŀ�꺯��ֵ������С����֮��ľ���
    distance[function1Ser.index[0]]=1000
    distance[function1Ser.index[-1]]=1000
    distance[function2Ser.index[0]]=1000
    distance[function2Ser.index[-1]]=1000

    ##�������������distanceֵ
    for i in range(1,len(Front)-1):
        distance[function1Ser.index[i]]=distance[function1Ser.index[i+1]]+(function1Ser[function1Ser.index[i-1]]-function1Ser[function1Ser.index[i-1]])/(max(function1_Front_List)-min(function1_Front_List))
        distance[function2Ser.index[i]]+=(function2Ser[function2Ser.index[i+1]]-function2Ser[function2Ser.index[i-1]])/(max(function2_Front_List)-min(function2_Front_List))

    distance.sort_values(ascending=False,inplace=True)
    print('distance is')
    print(distance)
    return distance##dataframe byte=float64


##���溯��
def crossover(ind1,ind2,n_param): #ģ������ƽ������Ӳ����Ӵ�
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

##���캯��
def gaussMutationStrength(generation): #����������쳤��
        random.seed()
        #return rd.normal(0, 0.2 * np.exp(-(1 - generation / generationMax)))
        return np.random.normal(0, 0.05 * (1 - generation / generationMax))

def mutate(child, generation): # �����Ӵ��ı��캯��
    random.seed()
    for i in range(8):
        if random.random() <= 0.1: # mutation probability
            child.params[i] = child.params[i] + gaussMutationStrength(generation) * (param_range[i][1] - param_range[i][0])

            child.params[i] = param_range[i][1] if child.params[i] > param_range[i][1] else child.params[i]
            child.params[i] = param_range[i][0] if child.params[i] < param_range[i][0] else child.params[i]

    return child

#�����б�ָ��Ԫ�ص�����,����a���б�list�������
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

##��values�б��еĸ����С�������в��ø���������ʾ
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        # ��������Ȳ����ڳ�ʼ����ʱ������ѭ��
        if index_of(min(values),values) in list1:##���values����б�����Сֵ��������list1��
            sorted_list.append(index_of(min(values),values))##��sorted_list�����Ӹ�����
        values[index_of(min(values),values)] = math.inf#ɾ��values�б��е���Сֵ
    print(sorted_list)
    return sorted_list



##������Ⱥ����
n_param=8
popSize=10
#����һ���������Ⱥ
population = createInitialPopulation(n_param,popSize)
print ('Complete initializing the population...')
##whileѭ��,�������������Ž�
time_start=time.time() #��¼ѭ����ʼʱ��
generationMax=1
generation = 1
while generation <= generationMax:##���ﵽ����ʱ��������������׼��û�����е�֧���ʱ���˳��Ż�����
    print ('Start optimization at generation %d...' % generation)
    function1_value_list=objectiveValueSet(population,popSize)[0]
    function2_value_list=objectiveValueSet(population,popSize)[1]

    non_dominated_sorted_solution=fast_non_dominated_sort(population,popSize,n_param)##��population���з�֧���ֲ�
    first_front=non_dominated_sorted_solution[0]##���������Ⱥ�������ŷ�֧���������
    population2=population##population2��ӵ��2��popSize����Ⱥ���Ƚ�����һ����帳ֵΪpopulation
    for j in range(0, popSize, 2):
        child1, child2 = crossover(population[j], population[j+1],n_param)##��population���ڵ��������彻���γ������µĸ���child1,child2
        mutate(child1, generation)##��child1��child2�ٽ��б����γ��µĸ���
        mutate(child2, generation)
        population2.addIndividual(child1)
        population2.addIndividual(child2)##��population2�м��뽻������γɵ��µĸ�������γɸ������Ӵ���ϵ���Ⱥpopulation2
    non_dominated_sorted_solution2=fast_non_dominated_sort(population2,popSize*2,n_param)##��population2���з�֧���ֲ�
    print('loop test')
    print('non_dominated_sorted_solution2')
    print(non_dominated_sorted_solution2)
    crowd_distance_front=[]##�����������ӵ�������б�
    crowding_distance_values2=[]#Ԫ��Ϊÿ��ӵ�������б���б�
    for i in range(0,len(non_dominated_sorted_solution2)):##������֧����е�ÿһ��
        ##distanceΪ��֧�������ӵ�����룬��Ϊ��data frame(byte=float64)��Ҫ����ת�����б�
        print(i)
        distance=crowdedDistance(population2,popSize*2,non_dominated_sorted_solution2[i])#.astype(pd.np.int64)##ת��float64
        print(distance)
        distance=distance.values##���data frame��ֵ
        print(distance)
        for j in distance:
            crowd_distance_front.append(j)
        crowding_distance_values2.append(crowd_distance_front)
        print('crowding_distance_values2')
        print(crowding_distance_values2)
    new_population_order= []##�����population2��ѡ����һ��������������
    for i in range(0,len(non_dominated_sorted_solution2)):##������֧����е�ÿһ��
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]##[0,1,2...]�иò�����Ԫ��
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])#�Ըò�����ӵ�����������ø���������ʾ����
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]#����������������population2��Ⱥ�е�����
        front.reverse()##���б������У�ʹӵ����������ʽ����ɴ�С
        for j in front:##����front������
            #��population2�����ǰ��һ������������ŷŵ��б�new_population_order.append(j)��
            #����ʽ���ȴ���ķ�֧���ѡȡ����ѡ���ٽ��֧��㣬������������popSize,���������ӵ��������ѡ��
            new_population_order.append(j)
            if(len(new_population_order)==popSize):
                break
        if (len(new_population_order) == popSize):
            break
    print('new_population_order')
    print(new_population_order)
    population =createInitialPopulation(n_param,popSize=100)##��������һ����Ⱥ
    for i,j in zip(new_population_order,range(0,popSize)):
        population.__setitem__(j,population2[i])##�������Ⱥ�ĸ��帳ֵΪpopulation2�е��������
    generation+=1
    print('generation is',generation)
print ('total cpu time: %7.4f s' % (time.time() - time_start))


#������
columns=['a1','a2', 'a3', 'a4','b1','b2', 'c1', 'c2',]
outputResult = pd.DataFrame(np.zeros((len(first_front), n_param+1)), columns=columns)
for i in range(len(first_front)):
    outputResult.iloc[i, 1:] = population[first_front[i]].params

outputResult.to_excel('optimization_result.xlsx')

##��ͼ
Y= [i for i in function1_value_list]##����Ч��
X= [k for k in function2_value_list]##�ɱ�
plt.xlabel('cost', fontsize=15)
plt.ylabel('environment', fontsize=15)
plt.scatter(X, Y)
plt.show()

