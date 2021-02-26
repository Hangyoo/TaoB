#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 09:25:40 2017
Tan Y, Zhu Y. Fireworks algorithm for optimization[C]// 
International Conference on Advances in Swarm Intelligence. Springer-Verlag, 2010:355-364.
@author: star
"""

from fireWorks_20210210.FireWork import FireWork
from fireWorks_20210210.ObjFunction import ObjFunc
import copy
import random
import matplotlib.pyplot as plt

class Population:
    """
    population
    """
    epsino=1e-30
    
    def __init__(self,PopSize,m_para,a_para,b_para,A_para,mm_para,RealBound,InitBound,vardim,FunName):
        self.m=m_para
        self.a=a_para
        self.b=b_para
        self.A=A_para
        self.mm=mm_para
        self.RealBound=RealBound
        self.InitBound=InitBound
        self.popsize=PopSize
        self.vardim=vardim
        self.pop=[]
        self.MaxFitness=0.
        self.MinFitness=0.
        self.FunName=FunName

        self.ID_rest = None
        
    def inilitialize(self):
        for i in range(0,self.popsize):
            ind=FireWork(self.vardim,self.InitBound)
            self.ID_rest = ind.initialize()
            ind.evaluate(self.FunName)
            self.pop.append(ind)#给self.pop列表里追加ind元素
            
    def UpdateMaxFintess(self):
        self.MaxFitness=self.pop[0].fitness
        for i in range(1,self.popsize):
            if self.MaxFitness<self.pop[i].fitness:
                self.MaxFitness=self.pop[i].fitness
    def UpdateMinFitness(self):
        self.MinFitness=self.pop[0].fitness
        for i in range(1,self.popsize):
            if self.MinFitness>self.pop[i].fitness:
                self.MinFitness=self.pop[i].fitness
    def CalculateSi(self):
        """
        Refer to equation (3)
        """
        self.UpdateMaxFintess()
        temp=0.
        for i in range(0,self.popsize):
            temp=temp+self.MaxFitness-self.pop[i].fitness
        for i in range(0,self.popsize):
            self.pop[i].si=self.m*(self.MaxFitness-self.pop[i].fitness+self.epsino)/(temp+self.epsino)
            if self.pop[i].si<self.a*self.m:
                self.pop[i].si=round(self.a*self.m)
            elif self.pop[i].si>self.b*self.m:
                self.pop[i].si=round(self.b*self.m)
            else:
                self.pop[i].si=round(self.pop[i].si)
    def CalculateExpo(self):
        """
        Refer to equation (4)
        """
        self.UpdateMinFitness()
        temp=0.
        for i in range(0,self.popsize):
            temp=temp+self.pop[i].fitness-self.MinFitness
        for i in range(0,self.popsize):
            self.pop[i].Ai=self.A*(self.pop[i].fitness-self.MinFitness+self.epsino)/(temp+self.epsino)
    
    def Explosion(self):
        """
        Refer to Algorithm 1 in the orginal paper
        """
        for k in range(0,self.popsize):
            spark_before = copy.deepcopy(self.pop[k])
            spark_later = copy.deepcopy(self.pop[k])
            rand = random.random()
            for j in range(int(self.pop[k].Ai)):
                # 替换为更好基因
                if rand > 0.5:
                    val_beform = ObjFunc(self.vardim, spark_before.location, "discrete")
                    id_new = random.choice(self.ID_rest)
                    idx = random.randint(0, 29)
                    while id_new in spark_later.location:
                        id_new = random.choice(self.ID_rest)
                    spark_later.location[idx] = id_new
                    val_later = ObjFunc(self.vardim, spark_later.location, "discrete")
                    if val_later < val_beform:
                        self.pop.append(spark_later)
                else:
                    # 随机替换基因
                    id_new = random.choice(self.ID_rest)
                    idx = random.randint(0, 29)
                    while id_new in spark_before.location:
                        id_new = random.choice(self.ID_rest)
                    spark_before.location[idx] = id_new
                    self.pop.append(spark_before)
        
    def Mutation(self):
        """
        Refer to Algorithm 2 in the orginal paper
        """
        for k in range(0,self.popsize):
            newpop=[]
            for i in range(0,self.pop[k].si):
                spark=copy.deepcopy(self.pop[k])
                #todo 交叉、互换、shuff
                p = random.random()
                if p <= 1/3:
                    num = 0
                    while num < 5:
                        id_new = random.choice(self.ID_rest)
                        idx = random.randint(0,29)
                        while id_new in spark.location:
                            id_new = random.choice(self.ID_rest)
                        spark.location[idx] = id_new
                        num += 1
                elif 1/3 < p <= 2/3:
                    pos1,pos2 = random.choices([i for i in range(30)],k=2)
                    spark.location[pos1], spark.location[pos2] = spark.location[pos2], spark.location[pos1]
                else:
                    random.shuffle(spark.location)
                spark.evaluate(self.FunName)
                newpop.append(spark)
            self.pop+=newpop
            
    def FindBest(self):
        index=0
        currentsize=len(self.pop)
        for i in range(1,currentsize):
            if self.pop[i].fitness<self.pop[index].fitness:
                index=i
        return index
    
    def Selection(self):
        newpop=[]
        newpop.append(self.best)
        for i in range(0,len(self.pop)):
            dis=0.
            for j in range(0,len(self.pop)):
                dis+=self.pop[i].distance(self.pop[j])
            self.pop[i].Ri=dis
        sr=0.
        for i in range(0,len(self.pop)):
            sr+=self.pop[i].Ri
        px=[]
        sum1=0.
        for i in range(0,len(self.pop)):
            sum1+=self.pop[i].Ri/sr
            px.append(sum1)
        for i in range(0,self.popsize):
            rr=random.uniform(0,1)
            index=0
            for j in range(0,len(self.pop)):
                if j==0 and rr<px[j]:
                    index=j
                elif rr>=px[j] and rr<px[j+1]:
                    index=j+1
            newpop.append(self.pop[index])
        self.pop=newpop

    def Run(self,MaxEva):
        self.inilitialize()
        bestindex=0
        self.best=copy.deepcopy(self.pop[bestindex])
        bx=[]
        e=0
        while FireWork.EvaNum<MaxEva:
            self.CalculateSi()
            self.CalculateExpo()
            self.Explosion()
            self.Mutation()
            bestindex=self.FindBest()
            if self.best.fitness>self.pop[bestindex].fitness:
                self.best=copy.deepcopy(self.pop[bestindex])
            self.Selection()
            print("Current pop size is %d Evaluation time is %d best fitness is %f"%(len(self.pop),FireWork.EvaNum,self.best.fitness))
            if self.best.fitness<self.epsino:
                break 
            bx.append(self.best.fitness) 
            e+=1        
        print("Best fitness %f"%self.best.fitness)
        print(self.best.location)
        x=[]
        y=[]
        for i in range(0,e):
            x.append(i)
            y.append(bx[i])
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.plot(x,y)
        plt.show()  
       

        
      

if __name__=="__main__":
    realbound=[0,1000]
    initbound=[0,1000]
    func="discrete"
    n=9
    m=50
    a=0.04
    b=0.8
    A_=40
    m_=5
    vardim=30
    maxeva=10000
    FAW=Population(n,m,a,b,A_,m_,realbound,initbound,vardim,func)
    FAW.Run(maxeva)
     
                    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        