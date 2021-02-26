#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 07:26:28 2017

@author: star
"""

"""
refer to http://www.cnblogs.com/biaoyu/p/4857881.html
"""

from fireWorks_20210210.ObjFunction import ObjFunc
import random
import math

class FireWork:
    """
    Single firework
    """
    EvaNum=0
    def __init__(self, vardim, bound):
        self.vardim=vardim
        self.bound=bound
        self.fitness=0.
        self.si=0.
        self.Ai=0.
        self.Ri=0.
        
    def initialize(self):
        """
        Initialization
        """
        len=self.vardim
        #长度为30的离散编码（不考虑标记）
        # temp = []
        # ID = [i for i in range(1000)]
        # for i in range(len):
        #     id = random.choice(ID)
        #     while id in temp:
        #         id = random.choice(ID)
        #     temp.append(id)
        # self.location=temp

        # 长度为30的离散编码（考虑标记）
        temp = []
        ID = [i for i in range(1000)]
        ID_mark = random.choices(ID,k=200)
        ID_rest = []
        for i in ID:
            if i not in ID_mark:
                ID_rest.append(i)
        for i in range(len):
            id = random.choice(ID_rest)
            while id in temp:
                id = random.choice(ID_rest)
            temp.append(id)
        self.location=temp
        return ID_rest

    def evaluate(self,FuncName):
        """
        Get fitness
        """
        self.fitness=ObjFunc(self.vardim,self.location,FuncName)
        FireWork.EvaNum+=1
        
    def distance(self,other):
        dis=0.
        for i in range(0,self.vardim):
            dis+=(self.location[i]-other.location[i])**2
        return math.sqrt(dis)
   