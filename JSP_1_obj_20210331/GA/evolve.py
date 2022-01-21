import random
import numpy as np

#选择操作
def select(Chrom,temp,GGAP):

    #初始化
    NIND, WNumber = Chrom.shape

    retain_length = int(NIND*GGAP)    #计算留下来的个体的数量
    elite = int(NIND-int(NIND*GGAP))  #精英数量

    ChromNew = np.zeros((retain_length,WNumber))
    Chromelite = np.zeros((elite, WNumber))

    # 将temp中，按适应值排序

    temp = sorted(temp, key=lambda x: x[0])  # 根据适应度值对列表进行排序
    inid_sort = [x[1] for x in temp]  # 根据个体适应度值的高低，对个体进行提取

    retain = inid_sort[0:retain_length]  # 根据保留比例GGAP，选择优秀个体
    retain_elite = inid_sort[0:elite]    #  据保留比例1-GGAP精英个体

    for i in range(retain_length):  # 得到新种群
        ChromNew[i,:] = retain[i]

    for i in range(elite):
        Chromelite[i,:] = retain_elite[i]

    return ChromNew,Chromelite

#交叉操作
def across(Chrom,pc,PNumber):

    #新种群
    NIND,WNumber =  Chrom.shape   #NIND为种群个体数量，WNumber为个体长度（工件数*工序数）
    ChromNew = Chrom
    s11 = [0]*WNumber
    s22 = [0]*WNumber

    #选择随机交叉个体
    L1 = []
    for i in range(NIND):
        L1.append(i)
    random.shuffle(L1) #将个体序号编码打乱，便于随机选取交叉个体
    SelNum = L1


    L2 = []
    for i in range(1,PNumber+1):  #工件的序号要从1开始索引
        L2.append(i)
    random.shuffle(L2)
    SelPNumber = L2

    pos1 = random.randint(0,PNumber-1)
    pos2 = random.randint(0,PNumber-1)

    while pos1 == pos2:           #确保pos2 与 pos1 不相等  且要确保POS2>POS1
        pos2 = random.randint(0,PNumber-1)

    if pos2 > pos1:
        pass
    else:
        pos1,pos2 = pos2,pos1

    SelPNumber = SelPNumber[pos1:pos2]  #选择机器

    #交叉个体对的个数
    Num = int(NIND/2)*2
    for i in range(0,Num,2):
        #若pc大于随机数则进行交叉
        if pc >random.random():

            #选取交换的个体
            s1 = Chrom[SelNum[i],:]    #SelNUm 为打乱个体的序列
            s2 = Chrom[SelNum[i+1],:]

            for j in range(0,WNumber):   #循环WNumber次，确保对染色体是上的每一个基因位置都有扫描

                if s1[j] in SelPNumber:  #SelPNumber 为pos1：pos2的片段
                    s11[j] = s1[j]
                else:
                    s11[j] = -1            #即使不赋值为0，原值也是0

                if s2[j] in SelPNumber:
                    s22[j] = s2[j]
                else:
                    s22[j] = -1


            for j in range(WNumber):
                k = 0
                if s2[j] not in SelPNumber:   #若s2(j)不在selPNUMBer中， 按顺序将S2(j)放入s11=0 的位置中
                    while s11[k]!=-1:
                        k += 1
                    s11[k] = s2[j]

                t = 0
                if s1[j] not in SelPNumber:
                    while s22[t] != -1:
                        t += 1
                    s22[t] = s1[j]

            #将交叉过后的新种群放入种群Chrom中
            ChromNew[SelNum[i],:] = s11
            ChromNew[SelNum[i+1],:] = s22
    return ChromNew

#变异操作(在染色体上任选两个位置进行互换)

def mutation(Chrom,pm):
    ChromNew = Chrom
    NIND, WNumber = Chrom.shape
    for i in range(NIND):
        #若变异概率大于随机数，那么进行变异
        if pm > random.random():
            s = Chrom[i,:]   #在种群中选取一个个体s进行变异
            pos1 = random.randint(0,WNumber-1)
            pos2 = random.randint(0,WNumber-1)

            while pos1 == pos2:
                pos2 = random.randint(0,WNumber-1)

            s[pos1],s[pos2] = s[pos2],s[pos1]

            ChromNew[i,:] = s
    return ChromNew


