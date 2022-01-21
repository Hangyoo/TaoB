import numpy as np
import fileinput
import pickle

#功能说明：根据个体S,计算调度工序P
def calp(s,PNumber):
    '''s为个体，PNumber为工件数量，输出参数P:为输出的调度工序 比如数字102表示工件10的工序3'''
    WNumber = len(s)
    MNumber = WNumber/PNumber   #每个工件的工序数量(机器个数)=总工序数/工件数

    if MNumber != int(MNumber):
        print('输入错误')
    else:
        pass

    #初始化
    temp = [0]*PNumber   #工件数量
    P = [0]*WNumber      #工序数量

    #编码生成调度工序：
    for i in range(WNumber):
        P[i] = s[i]*10 + temp[int(s[i])-1]   #索引从0开始，而S[i]的话从1开始，所以要减1

        #工序加1
        temp[int(s[i])-1] += 1
    return P

#根据调度工序,计算出调度工序时间,输出参数PVal  为调度工序开始加工时间及完成时间
def caltime(P,T,Jm):
    PNumber, MNumber = T.shape
    WNumber = PNumber*MNumber  #总工序数量
    if WNumber != len(P):
        print('输入错误')

    #初始化：
    TM = [0]*MNumber             #机器开始加工时间
    TP = [0]*PNumber             #前一工序的完工时间
    PVal = np.zeros((2,WNumber))   #用来存储每个工序的开工时间和完工时间

    #计算每一个工序的开工时间和完工时间：
    for i in range(WNumber):

        #取机器号
        value = P[i]   #对编码的P进行解码，得到工件号和工序号
        a = int((value % 10) +1)    #工序号  1开始   begin,end 为浮点数，不能作为索引
        b = int((value - a + 1)/10)  #工件号  1开始
        if b == 21:
            b = 20
        m = Jm[b-1,a-1]         #机器号(工件号，工序号)   索引从0开始.

        #取加工时间
        t = T[b-1,a-1]

        #取机器加工本工序的开始时间和前面一道工序的完成时间
        TMval = TM[m]
        TPval = TP[b-1]

        #机器加工本工序的开始时间 大于前面一道工序的完成时 ，取机器加工本工序的开始时间
        if TMval >TPval:
            nowtime = TMval    #当前工件的开始加工时间取机器完工时间
        else:
            nowtime = TPval

        #将完工时间填入列表中
        PVal[0,i] = nowtime      #工序的开工时间
        PVal[1,i] = nowtime + t  #工序完工时间

        #记录本次工序的机器时间和工序时间
        TP[b-1] = PVal[1,i]
        TM[m] = PVal[1,i]
    return PVal


#计算种群中每个个体的完工时间以及作业顺序，返回最优的顺序和完工时间
def cal(Chrom,T,Jm):   #Chrom为种群，T为加工时间矩阵，Jm为机器矩阵
    '''输出参数: PVal      为最佳调度工序时间
               P         为最佳输出的调度工序
               ObjV      为群中每个个体的调度工序时间'''
    #初始化：
    PNumber = T.shape[0]
    NIND= Chrom.shape[0]
    objV = [0]*NIND
    temp = []
    val1 = 0
    val2 = 0
    MinVal = 1000000000  #用来储存最小完工时间
    for i in range(NIND):
        s = list(Chrom[i,:]) #取一个个体
        P = calp(s,PNumber)         #根据基因，计算调度工序
        PVal = caltime(P,T,Jm)      #根据调度工序，计算出调度工序时间
        Cmax = PVal.max()          #PVal是一个两行多列array，TVal 为个体的完工时间
        objV[i] = Cmax              #保存每个个体的完工时间的列表
        temp.append((Cmax,s))       #(适应度值，个体)

        val1 = PVal
        val2 = P   #val2 用来储存加工次序

        #记录 最小的调度工序时间、最佳调度工序时间 最佳输出的调度工序
        if MinVal > Cmax:   #若大，则取小的
            MinVal = Cmax
            val1 = PVal
            val2 = P
        else:                  #现在个体的MINval比较好，什么也不做，继续比较个体
            pass

    #输出最佳调度工序时间PVal, 最优个体P
    PVal = val1
    P = val2

    return PVal,P,objV,temp

def makespan(P,T,Jm):   #Chrom为种群，T为加工时间矩阵，Jm为机器矩阵
    '''输出参数: TVal      为个体完工时间时间'''

    #初始化：
    PNumber = T.shape[0]
    PVal = caltime(P,T,Jm)      #根据调度工序，计算出调度工序时间
    TVal = PVal.max()          #PVal是一个两行多列array，TVal 为个体的完工时间
    return int(TVal)

def decode(P,T):
    PNumber, MNumber = T.shape
    WNumber = PNumber * MNumber
    individual = []
    for i in range(WNumber):
        value = P[i]
        a = int((value % 10) + 1)     # 工序号
        b = int((value - a + 1) / 10) #工件号
        individual.append(b)
    return individual



def readJobs(path=None):
    '''函数功能：读取文件转换为ndarray对象'''
    makespan = -1
    Jm = []
    T = []
    with fileinput.input(files=path) as f:
        jobs = [[(int(machine), int(span)) for machine, span in zip(*[iter(line.split())]*2)]
                    for line in f if line.strip()]
        '''jobs = [[(6,6)],
                  [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)], 
                  [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)], 
                  [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)], 
                  [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)], 
                  [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)], 
                  [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)]]'''
        for i in range(len(jobs)):
            if i ==0:
                n,m = jobs[0][0]  #获取工件数量和机器数量
                makespan = None
            else:
                for pair in jobs[i]:
                    Jm.append(pair[0])
                    T.append(pair[1])
        Jm = np.array(Jm).reshape((n,m))
        T = np.array(T).reshape((n, m))
    return Jm,T,makespan

def save_rest(bestList, path):
    with open(path, 'wb') as f:
        #pickle.dump((self.gbest, self.gbest_solution), f)
        pickle.dump(bestList, f)

if __name__ == '__main__':
    Jm = np.array([[2, 0, 1, 3, 5, 4],
                   [1, 2, 4, 5, 0, 3],
                   [2, 3, 5, 0, 1, 4],
                   [1, 0, 2, 3, 4, 5],
                   [2, 1, 4, 5, 0, 3],
                   [1, 3, 5, 0, 4, 2]])
    T = np.array([[1, 3, 6, 7, 3, 6], [8, 5, 10, 10, 10, 4], [5, 4, 8, 9, 1, 7], [5, 5, 5, 3, 8, 9],
                  [9, 3, 5, 4, 3, 1], [3, 3, 9, 10, 4, 1]])
    s = [3, 1, 4, 6, 6, 1, 2, 1, 3, 1, 4, 4, 3, 4, 3, 2, 6, 1,
         1, 5, 5, 2, 4, 3, 2, 6, 5, 6, 5, 4, 2, 3, 5, 6, 5, 2]
    P = calp(s,6)
    makespan = makespan(P,T,Jm)
    print(makespan)

    W = [20.0, 30.0, 60.0, 31.0, 10.0, 40.0, 61.0, 21.0, 50.0, 51.0, 11.0, 22.0, 41.0, 52.0, 32.0, 33.0, 62.0, 42.0, 43.0, 23.0, 34.0, 44.0, 12.0, 53.0, 13.0, 35.0, 14.0, 63.0, 24.0, 54.0, 64.0, 45.0, 15.0, 25.0, 65.0, 55.0]
    q = decode(W,T)
    print(q)



