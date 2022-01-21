__author__ = 'lly'
# [Purpose]
# [Version]
# SWMM 5.1.013
# [Language]
# Python 3.x
# [Commands]
# Note:swmm5.exe, swmm5.dll, SWMM_Model.py, and the SWMM inp file should be put in the same folder
# [Date]
# 2020-02-24
# [Author]
# Linyuan Leng
# Ph.D Candidate
# Environment School
# Tsinghua University
# Beijing, China
# Email: lengly18@mails.tsinghua.edu.cn
# Phone: 86-18801414367
# -*- coding: UTF-8 -*-
import random
import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import os
import re
import torch
import torch.nn as nn
import numpy as np
from numpy import random as rd
import pandas as pd
import time
from copy import deepcopy
import bisect

#--------------------------读取原始pipe/subcatchment/node信息-------------------#
##### dealing with the node tree
node_set = {}
pipe_set = {}
subc_set = {}

class PIPE:
    def __init__(self, name, upnode, downnode, length):
        self.name = name
        self.upnode = node_set[upnode] # Inlet Node 起点
        self.downnode = node_set[downnode] # Outlet Node 终点
        self.length = length
        self.D = 200.0 # 初始管径设为400mm
        self.Q = 0#设计流量
        self.v = 0#流速
        self.i = 0
        #第一个雨水口到pipe终点node的管内雨水流行时间
        self.t2 = 0
        self.offset_out = None # Outlet Offset
        self.offset_in = 0.0 # Inlet Offset

class NODE:
    def __init__(self, name, dem, x, y):
        self.name = name
        self.dem = dem
        self.xcor = x
        self.ycor = y
        self.invert = None # invert EI.
        self.depth = None # Max. Depth
        self.leaves = []#上游node集合(涉及的上游node有很多）
        self.root = None#下游node(下游node只有一个)
        self.downpipe = None#下游pipe
        self.subc = []#承担的子汇水区名称列表
        self.subcQ = 0
        self.t2 = 0 # t = t1 + m * t2
        #t=设计降雨历时=集水时间（汇水面积最远点到达该node时间）
        #t1:地面集水时间（汇水面积最远点到达第一个雨水口时间）经验值：10min
        #t2:第一个雨水口到该node的管内雨水流行时间
        #m:折减系数 管道取2

    #增加上游node，是一个list
    def add_leafnode(self, leafnode):
        self.leaves.append(node_set[leafnode])

    #设置下游node
    def set_rootnode(self, rootnode):
        self.root = node_set[rootnode]

    #定义下游pipe
    def set_downpipe(self, downpipe):
        self.downpipe = pipe_set[downpipe]

    #管网承担的子汇水区列表
    def add_subc(self, subc_list):
        self.subc.extend(subc_list)

class SUBC:
    def __init__(self, name, area_t, outlet, slope, rdsq_in, roof, green, rdsq_out, x, y):
        self.name = name
        self.area_t = area_t#面积m2
        self.outlet = outlet # Outlet 排水对应雨水检查井
        self.slope = slope # %Slope
        self.rdsq_in = rdsq_in
        self.roof = roof
        self.green = green
        self.rdsq_out = rdsq_out
        self.xcor = x
        self.ycor = y
        self.C = None
        self.P = None
        self.Q = None

    def calculate_runoff(self, p, t):
        return 0

#读取node信息(name,x,y,dem)
node_info = pd.read_excel('node.xlsx')
N_node = node_info.shape[0]#node数量
#对于node_set字典，键是node的名称，值是node类（目前拥有4个属性：name,x,y,dem)
for i in range(N_node):
    node_set[node_info.node_name[i]] = NODE(node_info.node_name[i], node_info.dem[i], node_info.x_cor[i], node_info.y_cor[i])

#读取pipe信息(name，起点node,终点node，长度）
pipe_info = pd.read_excel('pipe.xlsx')
N_pipe = pipe_info.shape[0]#pipe数量

#pipe_set字典，键是pipe名称，值是pipe类
# 根据读入数据确定4个属性：名称，起点node，终点node,长度
#根据pipe的信息为node类确定leafcode,rootcode,downpipe
for i in range(N_pipe):
    pipe_set[pipe_info.link_name[i]] = PIPE(pipe_info.link_name[i], pipe_info.upnode[i], pipe_info.downnode[i], pipe_info.length[i])
    #对这个管网的终点node类，把管网的起点node加入该node类的leafnode列表
    node_set[pipe_info.downnode[i]].add_leafnode(pipe_info.upnode[i])
    #对这个管网的起点node类，把管网的终点node加入该node类的root
    node_set[pipe_info.upnode[i]].set_rootnode(pipe_info.downnode[i])
    #对这个管网的起点node类，他的下游pipe就是这个管网
    node_set[pipe_info.upnode[i]].set_downpipe(pipe_info.link_name[i])
    # print(pipe_info.upnode[i])
    # print(node_set[pipe_info.upnode[i]].downpipe)

#读取subcatchment信息(name,outlet,不同地块面积，总面积坡度,x,y)
subc_info = pd.read_excel('subcatchment.xlsx')
N_subc = subc_info.shape[0]#汇水区数量

#subc_set类，键是sub名称，值是sub类
#根据读入数据确定属性：name，总面积,outlet,slope,不同地块面积，x,y)
for i in range(N_subc):
    subc_set[subc_info.subc_name[i]] = SUBC(subc_info.subc_name[i], subc_info.total_area[i], subc_info.outlet[i], subc_info.slope[i],\
                                            subc_info.rdsq_in[i], subc_info.roof[i], subc_info.green[i], subc_info.rdsq_out[i],\
                                            subc_info.X_center[i], subc_info.Y_center[i])
    #对这个子汇水的outlet node类，将该子汇水区加入node的subc列表
    node_set[subc_info.outlet[i]].add_subc([subc_set[subc_info.subc_name[i]]])

#----------------------------估计初始管径---------------------------------------------------#
##### estimate the pipe diameters
#雨水管网水力计算
#管网最大流速，最小流速
v_max = 4.0
v_min = 0.75
#坡度i（%）
i_max = 0.5
i_min = 0.1
#管径备选值
#主干管最小管径是400
D_list = [200.0,300.0,400.0, 450.0, 500.0, 600.0, 700.0, 800.0, 1000.0, 1200.0]

#计算设计降雨强度
#T：重现期（年）
#10+2t2=降雨持续时间
#t2管内雨水流行时间
def rainfall_intensity(T, t2):
    P = 3306.63 * (1 + 0.8201 * math.log(T, 10)) / ((10 + 2 * t2 + 18.99) ** 0.7735) # L/s*ha
    return P

#曼宁公式
def manning_eq_re(Q, D):
    #流速=设计流量/管道横截面积
    #D--mm
    v = Q / (0.25 * math.pi * (D / 1000.0) ** 2) # m/s
    #根据曼宁公式导坡度i
    #曼尼系数n取0.013
    i = (v * 0.013 / ((D / 1000.0 / 2) ** (2.0/3.0))) ** 2 * 100
    return v, i

#估计管径D
def estimate_pipe_d(node_object):
    #对该node对应的子汇水区
    for subc_object in node_object.subc:
        #计算子汇水区设计降雨强度
        #重现期=3年
        subc_object.P = rainfall_intensity(0.2, node_object.t2)
        #设计流量=降雨强度×径流系数×汇水区面积
        subc_object.Q = subc_object.P * subc_object.C * subc_object.area_t / 10000.0 / 1000.0 # m3/s
    # print(node_object.name)
    # print(node_object.downpipe)
    #该node下游pipe的设计流量
    node_object.downpipe.Q = node_object.downpipe.Q - node_object.subcQ + sum([x.Q for x in node_object.subc])
    #该node接受的所有子汇水区的设计流量
    node_object.subcQ = sum([x.Q for x in node_object.subc])
    #根据下游pipe的设计流量Q和管径D计算流速v和坡度i
    node_object.downpipe.v, node_object.downpipe.i = manning_eq_re(node_object.downpipe.Q, node_object.downpipe.D)
    d_i = 0#最小管径列表编号
    #初始管径设定400mm
    #如果v和i都大于最小值约束
    if node_object.downpipe.v >= v_min and node_object.downpipe.i >= i_min:
        #当v或i某一方超过最大约束并且编号小于7
        #如果没有超出约束，也就是v和i都在约束范围内，管径就不用增加，满足经济性要求
        while (node_object.downpipe.v > v_max or node_object.downpipe.i > i_max) and d_i < 7:
            d_i += 1#扩大管径
            node_object.downpipe.D = D_list[d_i]#更新管径
            #重新计算v和i
            node_object.downpipe.v, node_object.downpipe.i = manning_eq_re(node_object.downpipe.Q, node_object.downpipe.D)
    #管网的管内雨水流行时间=起点node的管内雨水流行时间+管道长度/流速
    node_object.downpipe.t2 = node_object.t2 + node_object.downpipe.length / node_object.downpipe.v
    #如果该node有下游node
    if node_object.root.root != None:
        #下游node的t2=下游pipe的t2
        node_object.root.t2 = node_object.downpipe.t2 if node_object.downpipe.t2 > node_object.root.t2 else node_object.root.t2
        #下游node的下游pipe设计流量=本段+Σ上游
        node_object.root.downpipe.Q += node_object.downpipe.Q
        #继续估计下游管道的管径
        #从起始雨水口node不断向下游估计，起始雨水口的t2=0
        estimate_pipe_d(node_object.root)

#估计径流系数
#subc_set字典
#subc_name键
#subc_object值
#道路屋顶0.9 绿地0.15
for subc_name, subc_object in subc_set.items(): # estimate the runoff coefficients
    subc_object.C = ((subc_object.rdsq_in + subc_object.rdsq_out + subc_object.roof) * 0.9 + subc_object.green * 0.15) / subc_object.area_t

for node_name, node_object in node_set.items():
    #如果该node没有上游node，作为起始雨水口不断向下游估计管径
    if len(node_object.leaves) == 0: # the beginning node
        estimate_pipe_d(node_object)

#检验？如果pipe的坡度i不在约束范围内，拉回范围内
for pipe_name, pipe_object in pipe_set.items():
    if pipe_object.i > i_max:
        pipe_object.i = i_max
    elif pipe_object.i < i_min:
        pipe_object.i = i_min
    else: pass

##------------------------计算初始埋深--------------------------------------------------#
##### calculate node inverts and outlet offsets
def calculate_leaf_invert(node_object):
    for leaf_node in node_object.leaves:
        #对该node的上游node
        try: # for middle nodes
            #下游pipe的offset_out=下游管径-上游管径
            leaf_node.downpipe.offset_out = (node_object.downpipe.D - leaf_node.downpipe.D) / 1000.0
        except: # for outlet nodes
            #outfall node的下游pipe，设下游pipe的坡度为1
            #outfall node的offset_out=0
            leaf_node.downpipe.i = 1.0
            leaf_node.downpipe.offset_out = 0
        #上游node的管底标高=下游node管底标高+上游node的下游pipe的offset_out+上游node的下游pipe的长度×坡度
        leaf_node.invert = node_object.invert + leaf_node.downpipe.offset_out +\
                           leaf_node.downpipe.length * leaf_node.downpipe.i / 100.0
        #上游node的埋深Max.depth=地面标高-管底标高
        leaf_node.depth = leaf_node.dem - leaf_node.invert
        #计算管底标高是从outfall往上游算，outfall的管底标高设为-1.5m
        calculate_leaf_invert(leaf_node)

#下面部分为计算埋深参数
for node_name, node_object in node_set.items():
    if node_object.root == None:
        #如果没有下游node，即outfall，设管底标高为-1.5m
        node_object.invert = -1.5
        #outfall的埋深为上游的最大管径
        node_object.depth = max([x.downpipe.D for x in node_object.leaves]) / 1000.0
        calculate_leaf_invert(node_object)

##-------------------------------------------生成inp文件--------------------------------------------------#
def inp(a1,a2,a3,a4,b1,b2,c1,c2,before_inp,after_inp):
    ### [SUBCATCHMENTS] Name Rain_Gage Outlet Area %Imperv Width %Slope "0"
    subcatchment_part = open("SUBCATCHMENTS.txt", 'w')
    ### [SUBAREAS] Subcatchment N-Imperv N-Perv S-Imperv S-Perv PctZero RouteTo PctRouted
    subarea_part = open("SUBAREA.txt", 'w')
    ### [INFILTRATION] Subcatchment MaxRate MinRate Decay DryTime MaxInfil
    infiltration_part = open("INFILTRATION.txt", 'w')
    ### [LID_USAGE] Subcatchment LID-abbr "1" "0.0010" Width "0" "0" "0"
    lid_usage_part = open("LID_USAGE.txt", 'w')
    ### [COVERAGES] Subcatchment Land_Use Percent
    coverage_part = open("COVERAGE.txt", 'w')
    ### [Polygons] Subcatchment X-Coord Y-Coord (four vertices for each)
    polygons_part = open("Polygons.txt", 'w')

    BR=0
    VS=0
    PP=0
    GR=0
    for subc_name, subc_object in subc_set.items():
        #建筑与小区（决策变量：A1、A2、A3、A4）
        if re.match("S[0-10_]+", subc_name) and subc_object.green<=subc_object.area_t*0.7 and subc_object.green>0.1:

            width = math.sqrt(subc_object.green) * 0.89
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_no', '1', subc_object.outlet, subc_object.green*(1-a1-a2)/ 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_BR', '1', subc_name+'_green_with_VS', subc_object.green*a1/ 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_VS', '1', subc_name+'_green_no', subc_object.green*a2 / 10000.0,\
                                                                                      0, width, subc_object.slope, 0))

            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_no', 0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_with_BR', 0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_with_VS',0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))

            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_no', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_with_BR', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_with_VS', 43.93, 0.38, 4.0, 9.0, 70))

            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_green_with_BR', 'BR', 1, int(subc_object.green*a1) , width, 0, 0, 0))
            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_green_with_VS', 'VS', 1, int(subc_object.green*a2) , width, 0, 0, 0))

            BR+=subc_object.green*a1
            VS+=subc_object.green*a2

            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_no', 'green', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_with_BR', 'green', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_with_VS', 'green', 100))

        #以GIS种地块中心x,y构建正方形地块
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+15, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+5, subc_object.ycor+15))

            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor+5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor+5, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor-5, subc_object.ycor+15))

            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-5, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-15, subc_object.ycor+15))

        #有RC的roof
        #无RC的roof
            if subc_object.roof >= 0.1:
                width = math.sqrt(subc_object.roof) * 0.31
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %d   %d\n" % (subc_name+'_roof_no', '1', subc_name+'_roof_with_GR', subc_object.roof*(1-a3) / 10000.0,\
                                                                                          100, width, 50, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %d   %d\n" % (subc_name+'_roof_with_GR', '1', subc_name+'_green_with_BR', subc_object.roof*a3/ 10000.0,\
                                                                                          100, width, 50, 0))

                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_roof_no', 0.024, 0.282, 0.3, 3.247, 0, 'OUTLET'))
                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_roof_with_GR', 0.024, 0.282, 0.3, 3.247, 0, 'OUTLET'))

                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_roof_no', 43.93, 0.38, 4.0, 9.0, 70))
                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_roof_with_GR', 43.93, 0.38, 4.0, 9.0, 70))

                lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_roof_with_GR', 'GR', 1, int(subc_object.roof*a3), width, 0, 0, 0))
                GR+=subc_object.roof*a3

                coverage_part.write("%s   %s   %d\n" % (subc_name+'_roof_no', 'roof', 100))
                coverage_part.write("%s   %s   %d\n" % (subc_name+'_roof_with_GR', 'roof', 100))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_no', subc_object.xcor-15, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_no', subc_object.xcor-5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_no', subc_object.xcor-5, subc_object.ycor+5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_no', subc_object.xcor-15, subc_object.ycor+5))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_with_GR', subc_object.xcor-5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_with_GR', subc_object.xcor+5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_with_GR', subc_object.xcor+5, subc_object.ycor+5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof_with_GR', subc_object.xcor-5, subc_object.ycor+5))

        #有pp的rdsq_in
        #无pp的rdsq_in
            if subc_object.rdsq_in >= 0.1:
                width = math.sqrt(subc_object.rdsq_in) * 1.27
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_in_no_PP', '1', subc_name+'_rdsq_in_with_PP', subc_object.rdsq_in*(1-a4) / 10000.0,\
                                                                                          100, width, subc_object.slope, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_in_with_PP', '1', subc_name+'_green_with_BR', subc_object.rdsq_in*a4/ 10000.0,\
                                                                                          100, width, subc_object.slope, 0))

                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_in_no_PP', 0.011, 0.282, 0.2, 3.247, 0, 'OUTLET'))
                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_in_with_PP', 0.011, 0.282, 0.2, 3.247, 0, 'OUTLET'))

                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_in_no_PP', 43.93, 0.38, 4.0, 9.0, 70))
                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_in_with_PP', 43.93, 0.38, 4.0, 9.0, 70))

                lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_rdsq_in_with_PP', 'PP', 1, int(subc_object.rdsq_in*a4) , width, 0, 0, 0))
                PP+=subc_object.rdsq_in*a4
                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_in_no_PP', 'rdsq_in', 100))
                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_in_with_PP', 'rdsq_in', 100))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_no_PP', subc_object.xcor+5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_no_PP', subc_object.xcor+15, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_no_PP', subc_object.xcor+15, subc_object.ycor+5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_no_PP', subc_object.xcor+5, subc_object.ycor+5))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_with_PP', subc_object.xcor+5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_with_PP', subc_object.xcor+15, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_with_PP', subc_object.xcor+15, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in_with_PP', subc_object.xcor+5, subc_object.ycor-5))
        #有pp的rdsq_out
        #无pp的rdsq_out
            if subc_object.rdsq_out >= 0.1:
                width = math.sqrt(subc_object.rdsq_out) * 9.59
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_no_PP', '1', subc_name+'_rdsq_out_with_PP', subc_object.rdsq_out*(1-a4)/ 10000.0,\
                                                                                          100, width, subc_object.slope, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_with_PP', '1', subc_name+'_green_with_VS', subc_object.rdsq_out*a4 / 10000.0,\
                                                                                          100, width, subc_object.slope, 0))

                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out_no_PP', 0.013, 0.282, 0.2, 3.247, 0, 'OUTLET'))
                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out_with_PP', 0.013, 0.282, 0.2, 3.247, 0, 'OUTLET'))

                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out_no_PP', 43.93, 0.38, 4.0, 9.0, 70))
                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out_with_PP', 43.93, 0.38, 4.0, 9.0, 70))

                lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_rdsq_out_with_PP', 'PP', 1, int(subc_object.rdsq_out*a4), width, 0, 0, 0))
                PP+=subc_object.rdsq_out*a4

                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out_no_PP', 'rdsq_out', 100))
                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out_with_PP', 'rdsq_out', 100))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no_PP', subc_object.xcor-5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no_PP', subc_object.xcor+5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no_PP', subc_object.xcor+5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no_PP', subc_object.xcor-5, subc_object.ycor-5))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-15, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-15, subc_object.ycor-5))

    #城市绿地（决策变量 B1、B2）
        elif re.match("S[0-10_]+", subc_name) and subc_object.green > subc_object.area_t*0.7:
            #print(subc_object.name)

            width = math.sqrt(subc_object.green) * 0.89
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_no', '1', subc_object.outlet, subc_object.green*(1-b1-b2)/ 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_BR', '1', subc_name+'_green_with_VS', subc_object.green*b1 / 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_VS', '1', subc_name+'_green_no', subc_object.green*b2 / 10000.0,\
                                                                                      0, width, subc_object.slope, 0))

            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_no', 0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_with_BR', 0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_green_with_VS',0.016, 0.282, 0.5, 3.247, 0, "OUTLET"))

            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_no', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_with_BR', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_green_with_VS', 43.93, 0.38, 4.0, 9.0, 70))

            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_green_with_BR', 'BR', 1, int(subc_object.green*b1) , width, 0, 0, 0))
            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_green_with_VS', 'VS', 1, int(subc_object.green*b2) , width, 0, 0, 0))
            BR+=subc_object.green*b1
            VS+=subc_object.green*b2
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_no', 'green', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_with_BR', 'green', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_green_with_VS', 'green', 100))

        #以GIS种地块中心x,y构建正方形地块
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+15, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_no', subc_object.xcor+5, subc_object.ycor+15))

            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor+5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor+5, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_BR', subc_object.xcor-5, subc_object.ycor+15))

            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-5, subc_object.ycor+15))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_green_with_VS', subc_object.xcor-15, subc_object.ycor+15))

        #有RC的roof
        #无RC的roof
            if subc_object.roof >= 0.1:
                width = math.sqrt(subc_object.roof) * 0.31
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %d   %d\n" % (subc_name+'_roof', '1', subc_object.outlet, subc_object.roof/ 10000.0,\
                                                                                          100, width, 50, 0))


                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_roof', 0.024, 0.282, 0.3, 3.247, 0, 'OUTLET'))

                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_roof', 43.93, 0.38, 4.0, 9.0, 70))

                coverage_part.write("%s   %s   %d\n" % (subc_name+'_roof', 'roof', 100))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof', subc_object.xcor-15, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof', subc_object.xcor-5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof', subc_object.xcor-5, subc_object.ycor+5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_roof', subc_object.xcor-15, subc_object.ycor+5))

        #有pp的rdsq_in
        #无pp的rdsq_in
            if subc_object.rdsq_in >= 0.1:
                width = math.sqrt(subc_object.rdsq_in) * 1.27
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_in', '1', subc_object.outlet, subc_object.rdsq_in/ 10000.0,\
                                                                                          100, width, subc_object.slope, 0))

                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_in', 0.011, 0.282, 0.2, 3.247, 0, 'OUTLET'))


                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_in', 43.93, 0.38, 4.0, 9.0, 70))


                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_in', 'rdsq_in', 100))


                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in', subc_object.xcor+5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in', subc_object.xcor+15, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in', subc_object.xcor+15, subc_object.ycor+5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_in', subc_object.xcor+5, subc_object.ycor+5))


        #有pp的rdsq_out
        #无pp的rdsq_out
            if subc_object.rdsq_out >= 0.1:
                width = math.sqrt(subc_object.rdsq_out) * 9.59
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out', '1', subc_object.outlet, subc_object.rdsq_out*0.8 / 10000.0,\
                                                                                          100, width, subc_object.slope, 0))

                subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out', 0.013, 0.282, 0.2, 3.247, 0, 'OUTLET'))

                infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out', 43.93, 0.38, 4.0, 9.0, 70))


                coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out', 'rdsq_out', 100))

                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out', subc_object.xcor-5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out', subc_object.xcor+5, subc_object.ycor-15))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out', subc_object.xcor+5, subc_object.ycor-5))
                polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out', subc_object.xcor-5, subc_object.ycor-5))

    #城市道路（决策变量C1、C2）
        else:
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_no','1', subc_name+'_rdsq_out_with_PP', (subc_object.rdsq_in + subc_object.rdsq_out)*(1-c1-c2)/ 10000.0,\
                                                                                    100, math.sqrt((subc_object.rdsq_in + subc_object.rdsq_out)) * 9.59, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_with_PP', '1', subc_name+'_rdsq_out_with_VS', (subc_object.rdsq_in + subc_object.rdsq_out)*c1/ 10000.0,\
                                                                                      100, math.sqrt((subc_object.rdsq_in + subc_object.rdsq_out)) * 9.59, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_with_VS', '1', subc_object.outlet, (subc_object.rdsq_in + subc_object.rdsq_out)*c2/ 10000.0,\
                                                                                      0, 5.0, subc_object.slope, 0))

            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out_no', 0.013, 0.282, 0.2, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out_with_PP',0.013, 0.282, 0.2, 3.247, 0, "OUTLET"))
            subarea_part.write("%s   %5.3f   %5.3f   %5.3f   %5.3f   %d   %s\n" % (subc_name+'_rdsq_out_with_VS',0.013, 0.282, 0.2, 3.247, 0, "OUTLET"))

            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out_no', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out_with_PP', 43.93, 0.38, 4.0, 9.0, 70))
            infiltration_part.write("%s   %5.2f   %5.2f   %5.1f   %d   %d\n" % (subc_name+'_rdsq_out_with_VS', 43.93, 0.38, 4.0, 9.0, 70))

            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_rdsq_out_with_PP', 'PP', 1, int((subc_object.rdsq_in + subc_object.rdsq_out)*c1), math.sqrt((subc_object.rdsq_in + subc_object.rdsq_out)) * 9.59, 0, 0, 0))
            lid_usage_part.write("%s   %s   %d   %7.4f   %6.3f   %d   %d   %d\n" % (subc_name+'_rdsq_out_with_VS', 'VS', 1, int((subc_object.rdsq_in + subc_object.rdsq_out)*c2), 5.0, 0, 0, 0))
            PP+=(subc_object.rdsq_in + subc_object.rdsq_out)*c1
            VS+=(subc_object.rdsq_in + subc_object.rdsq_out)*c2

            coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out_no', 'rdsq_out', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out_with_PP', 'rdsq_out', 100))
            coverage_part.write("%s   %s   %d\n" % (subc_name+'_rdsq_out_with_VS', 'green', 100))

            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no', subc_object.xcor-15, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no', subc_object.xcor-5, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_no', subc_object.xcor-15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-5, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor+5, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor+5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_PP', subc_object.xcor-5, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_VS', subc_object.xcor+5, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_VS', subc_object.xcor+15, subc_object.ycor-5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_VS', subc_object.xcor+15, subc_object.ycor+5))
            polygons_part.write("%s   %7.4f   %7.4f\n" % (subc_name+'_rdsq_out_with_VS', subc_object.xcor+5, subc_object.ycor+5))


    subcatchment_part.close()
    subarea_part.close()
    infiltration_part.close()
    lid_usage_part.close()
    coverage_part.close()
    polygons_part.close()

    ### [JUNCTIONS] Name Elevation MaxDepth "0" "0" "0"
    junction_part = open("JUNCTIONS.txt", 'w')
    ### [OUTFALLS] Name Elevation Type Stage_Data Gated Route_To
    outfall_part = open("OUTFALLS.txt", 'w')
    ### [COORDINATES] Node X-Coord Y-Coord
    coordinates_part = open("COORDINATES.txt", 'w')

    for node_name, node_object in node_set.items():
        if re.match('OUT', node_name):
            outfall_part.write("%s   %7.4f   %s   %7.4f   %s\n" % (node_name, node_object.invert, "FIXED", node_object.invert + node_object.depth, "YES"))
            coordinates_part.write("%s   %7.4f   %7.4f\n" % (node_name, node_object.xcor, node_object.ycor))

        else:
            junction_part.write("%s   %7.4f   %7.4f   %d   %d   %d\n" % (node_name, node_object.invert, node_object.depth, 0, 0, 0))
            coordinates_part.write("%s   %7.4f   %7.4f\n" % (node_name, node_object.xcor, node_object.ycor))

    junction_part.close()
    outfall_part.close()
    coordinates_part.close()

    ### [CONDUITS] Name From_Node To_Node Length Roughness InOffset OutOffset InitFlow MaxFlow
    conduit_part = open("CONDUITS.txt", 'w')
    ### [XSECTIONS] Link Shape Geom1 "0" "0" "0" "1"
    xsection_part = open("XSECTIONS.txt", 'w')

    for pipe_name, pipe_object in pipe_set.items():
        try:
            conduit_part.write("%s   %s   %s   %7.4f   %5.3f   %d   %3.2f   %d   %d\n" % (pipe_name, pipe_object.upnode.name, pipe_object.downnode.name,\
                                                                                  pipe_object.length, 0.013, pipe_object.offset_in, pipe_object.offset_out, 0, 0))
        except:
            print(pipe_name)
        xsection_part.write("%s   %s   %3.2f   %d   %d   %d   %d\n" % (pipe_name, "CIRCULAR", pipe_object.D / 1000.0, 0, 0, 0, 1))

    conduit_part.close()
    xsection_part.close()

    with open (before_inp) as f:
        lines=f.readlines()##文件内容读取列表

        a=open('SUBCATCHMENTS.txt')
        subcatchment=a.readlines()##文件内容读取列表
        subcatchment=" ".join(subcatchment)
        for line in lines:
            if line=='[SUBCATCHMENTS]\n':
                num_1=lines.index(line)
                lines.insert((num_1+3),subcatchment)

        b=open('SUBAREA.txt')
        subareas=b.readlines()##文件内容读取列表
        subareas=" ".join(subareas)
        for line in lines:
            if line=='[SUBAREAS]\n':
                num_2=lines.index(line)
                lines.insert((num_2+3),subareas)

        c=open('INFILTRATION.txt')
        infiltration=c.readlines()##文件内容读取列表
        infiltration=" ".join(infiltration)
        for line in lines:
            if line=='[INFILTRATION]\n':
                num_3=lines.index(line)
                lines.insert((num_3+3),infiltration)

        d=open('LID_USAGE.txt')
        lid_usage=d.readlines()##文件内容读取列表
        lid_usage=" ".join(lid_usage)
        for line in lines:
            if line=='[LID_USAGE]\n':
                num_4=lines.index(line)
                lines.insert((num_4+3),lid_usage)

        e=open('JUNCTIONS.txt')
        junction=e.readlines()##文件内容读取列表
        junction=" ".join(junction)
        for line in lines:
            if line=='[JUNCTIONS]\n':
                num_5=lines.index(line)
                lines.insert((num_5+3),junction)

        f=open('OUTFALLS.txt')
        outfall=f.readlines()##文件内容读取列表
        outfall=" ".join(outfall)
        for line in lines:
            if line=='[OUTFALLS]\n':
                num_6=lines.index(line)
                lines.insert((num_6+3),outfall)

        i=open('CONDUITS.txt')
        couduit=i.readlines()##文件内容读取列表
        couduit=" ".join(couduit)
        for line in lines:
            if line=='[CONDUITS]\n':
                num_7=lines.index(line)
                lines.insert((num_7+3),couduit)

        j=open('XSECTIONS.txt')
        xsection=j.readlines()##文件内容读取列表
        xsection=" ".join(xsection)
        for line in lines:
            if line=='[XSECTIONS]\n':
                num_8=lines.index(line)
                lines.insert((num_8+3),xsection)

        k=open('COVERAGE.txt')
        coverage=k.readlines()##文件内容读取列表
        coverage=" ".join(coverage)
        for line in lines:
            if line=='[COVERAGES]\n':
                num_9=lines.index(line)
                lines.insert((num_9+3),coverage)

        l=open('COORDINATES.txt')
        COORDINATES=l.readlines()##文件内容读取列表
        COORDINATES=" ".join(COORDINATES)
        for line in lines:
            if line=='[COORDINATES]\n':
                num_10=lines.index(line)
                lines.insert((num_10+3),COORDINATES)

        m=open('Polygons.txt')
        Polygons=m.readlines()##文件内容读取列表
        Polygons=" ".join(Polygons)
        for line in lines:
            if line=='[Polygons]\n':
                num_11=lines.index(line)
                lines.insert((num_11+3),Polygons)


    with open (after_inp,'w',encoding='utf-8') as f:
        for line in lines:
            f.write(line)

    return BR,VS,PP,GR

###----------------------------LSTM模型基本信息----------------------------###
#对于LSTM结构中的sigmoid函数和双曲正切函数，当输入过大时，如绝对值大于5时，函数梯度将趋近于零，即梯度消失，导致训练无法继续
# 为了尽可能避免梯度消失，
# 本研究进一步对每一维输入和输出使用所有训练样本中的最大值和最小值进行标准化处理：
label_max = np.array([154.57759094, 31.94049835, 2.89375615, 3.09694052, 2.7058022, 0.29921889, 0.18103579, 15.30073643])
label_min = np.array([6.55922532e-01, 1.00000005e-03, 4.88706753e-02, 1.11312442e-01, 4.99317702e-03, 1.55077281e-03, 2.56796228e-03, 2.29122591e+00])
dynamic_dist = False # 引水量是否动态调整

upper_limit = [2.00, 10, 10, 20, 1, 0.850, 0.10, 1.00]
lower_limit = [0.50, 0, 0, 2, 1, 0.000, 0.00, 0.00]
global_warming_factor = 1.0 # 气候变暖背景下底泥释放增强系数
# 0-夏季引水量, m3/s, [0.5, 2.0]
# 1-旁位处理, 万吨/天, {0,1,2,3,4,5,6,7,8,9,10}
# 2-原位净化, 万吨/天, {0,1,2,3,4,5,6,7,8,9,10}
# 3-底泥疏浚频率，年, {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
# 4-是否开展生态修复, {0,1}
# 5-绿色基础设施调控的综合目标, [0, 0.85]
# 6-夏季引水量的动态调控幅度, gamma, [0, 0.1]
# 7-夏季引水量的动态调控权重, weight, [0, 1]


S_T = 0.2066
S_N = 0.0008
S_P = 0.1442
S_log_V = 0.3765

rolling_w = 7 # 循环窗口规模
h_dim = 288 # LSTM隐含层规模
lstm_layer_n = 4  # LSTM层数

bagging_n = 5 # 学习器的个数
final_classifiers = [12, 14, 11, 2, 13] # Adaboost训练的强学习器索引
feature_n = 52 # 特征规模（所有列）
output_n = 8 # 输出规模

###-------------------------------模型结构-------------------------------###
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('%s loaded...' % device)

save_path = './results'

class Agent_LSTM(nn.Module):
    def __init__(self, x_dim, h_dim, num_layer):
        super(Agent_LSTM, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.lstm_layer = nn.LSTM(input_size=x_dim, hidden_size=h_dim, num_layers=num_layer, batch_first=True)
        self.fc = nn.Linear(self.h_dim, feature_n)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        y, _ = self.lstm_layer(x)  # dimension of out: [batch_size, seq, h_dim]
        out = y[:,-1]
        out = self.fc(out)

        return out

def model_simulation(input_df):
    y_pred = np.zeros((input_df.shape[0], output_n))#52×8 0数组

    # bagging的模拟结果
    for random_sd in range(bagging_n):
        y_temp = np.zeros((input_df.shape[0], output_n))#52×8 0数组
        #第一行标准化
        y_temp[0] = (input_df.iloc[0,:output_n].values - label_min) / (label_max - label_min)
        final_classifier = final_classifiers[random_sd]
        #调用训练模型
        net = torch.load('./强%d弱%d分类器.pt' % (random_sd, final_classifier), map_location='cpu')
        net.eval()

        # 第一个滚动窗口
        for t in range(1, rolling_w):
            init_wt = np.zeros((t, output_n))
            #8列 1-6行
            init_wt[0,:output_n] = y_temp[0] # replace the inititial value with the predicted value, y[0]
            #用标准化后的第一行替代

            features = np.hstack((init_wt, input_df.iloc[1:t+1,output_n:feature_n].values))
            features = np.vstack((np.zeros((rolling_w-t,feature_n)), features))
            features_t = torch.from_numpy(np.expand_dims(features, axis=0)) # 输入数据维度: [批量规模，滚动区间长度, 输入规模]
            features_t = features_t.float()

            output_t = net(features_t).detach().numpy()
            y_temp[t] = output_t[0]
            y_temp[t][output_t[0] < 0] = 0
            y_temp[t][output_t[0] > 1] = 1

        # 接下来的滚动窗口
        for t_ in range(1, input_df.shape[0]-rolling_w+1):
            t = t_ + rolling_w - 1
            init_wt = np.zeros((rolling_w, output_n))
            init_wt[0,:output_n] = y_temp[t_-1] # 用前期模拟结果代替本滚动区间的初始值，防止引入未来函数

            features = np.hstack((init_wt, input_df.iloc[t_:t_+rolling_w,output_n:feature_n].values))
            features_t = torch.from_numpy(np.expand_dims(features, axis=0))
            features_t = features_t.float()

            output_t = net(features_t).detach().numpy()
            y_temp[t] = output_t[0]
            y_temp[t][output_t[0] < 0] = 0
            y_temp[t][output_t[0] > 1] = 1

        y_pred += y_temp

    y_pred = y_pred / bagging_n
    #恢复归一化前数量级
    y_pred=y_pred* (label_max - label_min)+ label_min

    NH4N=y_pred[:,3].tolist()
    #TN：有机氮、氨氮、硝氮
    TN=y_pred[:,2]+y_pred[:,3]+y_pred[:,4].tolist()
    #TP：有机磷、磷酸盐
    TP=y_pred[:,5]+y_pred[:,6].tolist()

    a=sum([max((i-1),0) for i in NH4N])/sum([i for i in NH4N])
    b=sum([max((i-1),0) for i in TN])/sum([i for i in TN])
    c=sum([max((i-0.2),0) for i in TP])/sum([i for i in TP])

    results=(1/3)*a+(1/3)*b+(1/3)*c
    return results

###------------------------------更新输入文件-------------------------------###
def update_dataframe(df, r_gi):
    # 0-夏季引水量, m3/s, [0.5, 2.0]
    # 1-旁位处理, 万吨/天, {0,1,2,3,4,5,6,7,8,9,10}
    # 2-原位净化, 万吨/天, {0,1,2,3,4,5,6,7,8,9,10}
    # 3-底泥疏浚频率，年, {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}
    # 4-是否开展生态修复, {0,1}
    # 5-绿色基础设施调控的综合目标, [0, 0.85]
    #基于锁定效应的情景77确定除绿色设施外的参数，只更新径流和污染物的削减

    df_temp = deepcopy(df)

    ## 更新与夏季引水相关，此处还未考虑动态更新
    wt_max = 3
    #wt_p = features[0]
    wt_p = 0.5
    k_p = (wt_max - wt_p) / 15

    wt_p1 = [wt_max-k_p*t for t in range(15)]
    #wt_p2 = [features[0]] * 155
    wt_p2 = [0.5] * 155
    wt_p3 = wt_p1[::-1]
    wt_p1.extend(wt_p2)
    wt_p1.extend(wt_p3)

    df_temp['in_q'].iloc[147:332] = wt_p1

    ## 更新与旁位处理相关，此处还未考虑动态更新
    #features1_s = features[1] * 0.114 # * 10000 / (3600 * 24)
    features1_s = 5 * 0.114 # * 10000 / (3600 * 24)
    df_temp['by_treat_q'].iloc[1:] = features1_s

    #if features1_s > features[0]:
    if features1_s > 0.5:
        k_b = (features1_s - wt_p) / 15

        bt_p1 = [features1_s-k_b*t for t in range(15)]
        bt_p2 = [wt_p] * 155
        bt_p3 = bt_p1[::-1]
        bt_p1.extend(bt_p2)
        bt_p1.extend(bt_p3)

        df_temp['by_treat_q'].iloc[147:332] = bt_p1

    ## 更新与原位净化相关
    #df_temp['situ_treat_q'].iloc[1:] = features[2] * 0.114 # * 10000 / (3600 * 24)
    df_temp['situ_treat_q'].iloc[1:] = 2 * 0.114 # * 10000 / (3600 * 24)

    ## 更新与底泥疏浚相关- n, BFTN, BFNH4, BFPO4, SOD
    #r_sd = 1 - 2 * (0.5 ** (features[3] / 2) - 1) / (np.log(0.5) * features[3])
    r_sd = 1 - 2 * (0.5 ** (2 / 2) - 1) / (np.log(0.5) * 2)
    df_temp['n'].iloc[1:] = 0.025 + 0.01 * r_sd
    df_temp['SOD'].iloc[1:] = -3.15 * global_warming_factor * r_sd

    ## 更新与生态修复相关
    #df_temp['PR'].iloc[1:] = 0.05 if features[4] == 1 else 0.001
    df_temp['PR'].iloc[1:] = 0.05 if 1 == 1 else 0.001

    ## 更新与绿色基础设施相关- r_gi = (r_VOL, r_COD, r_DON, r_NH4, r_DOP, r_PO4)
    df_temp['runoff_q'] = df.runoff_q * (1 - r_gi[0])
    df_temp['runoff_TOC'] = df.runoff_TOC * (1 - r_gi[1])
    df_temp['runoff_DON'] = df.runoff_DON * (1 - r_gi[2])
    df_temp['runoff_NH4'] = df.runoff_NH4 * (1 - r_gi[3])
    df_temp['runoff_DOP'] = df.runoff_DOP * (1 - r_gi[4])
    df_temp['runoff_PO4'] = df.runoff_PO4 * (1 - r_gi[5])

    df_temp['BFCOD'].iloc[1:] = 7.68 * (1 - r_gi[1])
    df_temp['BFTN'].iloc[1:] = 0.026 * (1 - r_gi[2])
    df_temp['BFNH4'].iloc[1:] = 0.12 * global_warming_factor * r_sd + 0.059 * (1 - r_gi[3])
    df_temp['BFPO4'].iloc[1:] = 0.02 * global_warming_factor * r_sd + 0.006 * (1 - r_gi[5])

    return df_temp

class Model:
    def __init__(self,file_name):##类的名称属性
         self.file_name=file_name

    def run_model(self,file_name):##模型运行函数
          self.inp = file_name + '.inp'
          self.rpt = file_name + '.rpt'
          self.out = file_name + '.out'
          os.system(r'swmm5 %s %s %s' % (self.inp, self.rpt, self.out)) # run SWMM

    def get_runoff_results(self,file_name):
        Model.run_model(self,file_name)
        with open (file_name+'.rpt','r+',encoding='utf-8',errors='ignore') as f:
            simulationResults_list=f.readlines()
            global runoff,COD,TN,TP,NH4N
            for i in simulationResults_list:
                if 'Flow Routing Continuity' in i:
                    runoff=float(simulationResults_list[simulationResults_list.index(i)+7].split()[3])*10000##m3
                if 'Quality Routing Continuity' in i:
                    COD=float(simulationResults_list[simulationResults_list.index(i)+7].split()[3])##kg
                    TN=float(simulationResults_list[simulationResults_list.index(i)+7].split()[4])##kg
                    NH4N=float(simulationResults_list[simulationResults_list.index(i)+7].split()[5])##kg
                    TP=float(simulationResults_list[simulationResults_list.index(i)+7].split()[6])##kg
                    return runoff,COD,TN,TP,NH4N

    def get_flood_results(self,file_name):
        Model.run_model(self,file_name)
        with open (file_name+'.rpt','r+',encoding='utf-8',errors='ignore') as f:
            simulationResults_list=f.readlines()
            global flood
            for i in simulationResults_list:
                if 'Flow Routing Continuity' in i:
                    flood=float(simulationResults_list[simulationResults_list.index(i)+8].split()[3])*10000##m3
                    return flood

##确定决策变量范围
##建筑与小区（生物滞留设施率A1、植草沟率A2、绿色屋顶率A3、透水铺装率A4）
#城市绿地（绿地率>70%)（生物滞留设施率B1、植草沟率B2）
#城市道路（透水铺装率C1、植草沟率C2）

param_range = {\
               0:(0.01, 0.3),
               1:(0.01, 0.3),
               2:(0.01, 0.3),
               3:(0.01, 0.8),
               4:(0.01, 0.4),
               5:(0.01, 0.4),
               6:(0.01, 0.25),
               7:(0.01, 0.25),
               } # A1.A2.A3.A4.B1.B2.C1.C2

##input information


##Program
##定义多目标计算函数
##环境函数
def function1(a1,a2,a3,a4,b1,b2,c1,c2):
    #----------------------------得到径流数据-----------------------------#aa
    #生成给定决策变量的inp文件
    inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_3.inp','pjxc_3.inp')
    #生成SWMM模型
    M=Model('pjxc_3')
    runoff,COD,TN,TP,NH4N=M.get_runoff_results('pjxc_3')
    #基础情景径流量16999m3、COD 4284.65、TN 127.229、 NH4N 39.912、 TP 27.065
    C_runoff=(16999-runoff)/16999#径流削减率
    C_COD=(4284.65 -COD)/4284.65
    C_TN=(127.229 -TN)/127.229
    C_NH4N=(39.912 -NH4N)/39.912
    C_TP=(27.065-TP)/27.065
    #-------------------------------得到溢流数据--------------------------#
    #基础情景下10年溢流量 161 20年溢流量 284 30年溢流量 373 50年溢流量 499
    #inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_10.inp','pjxc_10.inp')
    #生成SWMM模型
    #M=Model('pjxc_10')
    #flood_10=M.get_flood_results('pjxc_10')
    #C_flood_10=(161-float(flood_10))/161#溢流削减率

    #inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_20.inp','pjxc_20.inp')
    #生成SWMM模型
    #M=Model('pjxc_20')
    #flood_20=M.get_flood_results('pjxc_20')
    #C_flood_20=(284-float(flood_20))/284#溢流削减率

    #inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_30.inp','pjxc_30.inp')
    #生成SWMM模型
    #M=Model('pjxc_30')
    #flood_30=M.get_flood_results('pjxc_30')
    #C_flood_30=(373-float(flood_30))/373#溢流削减率

    #inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_50.inp','pjxc_50.inp')
    #生成SWMM模型
    #M=Model('pjxc_50')
    #flood_50=M.get_flood_results('pjxc_50')
    #C_flood_50=(499-float(flood_50))/499#溢流削减率
    #C_flood=0.6*C_flood_10+0.2*C_flood_20+0.1*C_flood_30+0.1*C_flood_50

    #df_basis = pd.read_excel('基准情景.xlsx', skiprows=[0,2], index_col=0) # 优化程序也将调用，作为模板
    ## 更新与绿色基础设施相关- r_gi = (r_VOL, r_COD, r_DON, r_NH4, r_DOP, r_PO4)
    #r_gi=(C_runoff, C_COD, C_TN, C_NH4N, C_TP, C_TP)
    #input_df = update_dataframe(df_basis, r_gi)
    #C_river=(0.4323-model_simulation(input_df))/0.4323
    #C_envir=C_runoff+C_flood+C_river
    return C_runoff


##成本函数，LID设施的全生命周期成本
def function2(a1,a2,a3,a4,b1,b2,c1,c2):
    BR,VS,PP,GR=inp(a1,a2,a3,a4,b1,b2,c1,c2,'pjxc_blank_3.inp','pjxc_3.inp')
    t=1
    cost=BR*317+VS*120+PP*216+GR*200
    while t <=20:##绿色设施寿命
        cost+=(BR*317*0.07+VS*120*0.02+PP*216*0.05+GR*200*0.02)*(1/(1+0.035)**t)
        t+=1
    return cost

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

##定义一个个体，一个个体是一组决策变量
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
        print('population',population[i])
        FrontSet.append(population[i])
    ##保存这层个体的目标函数值
    print('FrontSet',FrontSet)
    function1_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[0]
    #print(function1_Front_List)
    function2_Front_List=objectiveValueSet(FrontSet,len(FrontSet))[1]

    function1Ser=pd.Series(function1_Front_List,index=Front)
    function2Ser=pd.Series(function2_Front_List,index=Front)

    ##目标函数值排序
    function1Ser.sort_values(ascending=False,inplace=True)
    function2Ser.sort_values(ascending=False,inplace=True)

    #print('function test')
    #print(function1Ser)
    #print(function2Ser)

##设置这层中目标函数值最大和最小个体之间的距离
    print(function1Ser)
    distance[function1Ser.index[0]]=1000
    distance[function1Ser.index[-1]]=1000
    distance[function2Ser.index[0]]=1000
    distance[function2Ser.index[-1]]=1000

    ##计算其他个体的distance值
    for i in range(1,len(Front)-1):
        distance[function1Ser.index[i]]=distance[function1Ser.index[i+1]]+(function1Ser[function1Ser.index[i-1]]-function1Ser[function1Ser.index[i-1]])/(max(function1_Front_List)-min(function1_Front_List))
        distance[function2Ser.index[i]]+=(function2Ser[function2Ser.index[i+1]]-function2Ser[function2Ser.index[i-1]])/(max(function2_Front_List)-min(function2_Front_List))

    distance.sort_values(ascending=False,inplace=True)
    #print('distance is')
    #print(distance)
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
    print(function1_value_list)
    print(function2_value_list)

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
    #print('loop test')
    #print('non_dominated_sorted_solution2')
    #print(non_dominated_sorted_solution2)
    crowd_distance_front=[]##储存这层个体的拥挤距离列表
    crowding_distance_values2=[]#元素为每层拥挤距离列表的列表
    for i in range(0,len(non_dominated_sorted_solution2)):##遍历非支配层中的每一层
        ##distance为该支配层个体的拥挤距离，因为是data frame(byte=float64)，要将其转换成列表
        #print(i)
        distance=crowdedDistance(population2,popSize*2,non_dominated_sorted_solution2[i])#.astype(pd.np.int64)##转换float64
        #print(distance)
        distance=distance.values##获得data frame的值
        #print(distance)
        for j in distance:
            crowd_distance_front.append(j)
        crowding_distance_values2.append(crowd_distance_front)
        #print('crowding_distance_values2')
        #print(crowding_distance_values2)
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
    #print('new_population_order')
    population =createInitialPopulation(n_param,popSize)##重新生成一个种群
    for i,j in zip(new_population_order,range(0,popSize)):
        population.__setitem__(j,population2[i])##将这个种群的个体赋值为population2中的优秀个体
    generation+=1
    print('generation is',generation)
print ('total cpu time: %7.4f s' % (time.time() - time_start))


#保存结果
columns=["envirgoal","cost",'a1','a2', 'a3', 'a4','b1','b2', 'c1', 'c2']
outputResult = pd.DataFrame(np.zeros((len(first_front), n_param+1)), columns=columns)
for i in range(len(first_front)):
    outputResult.iloc[i, "envirgoal"] = function1_value_list
    outputResult.iloc[i, "cost"] = function2_value_list
    outputResult.iloc[i, 2:] = population[first_front[i]].params


outputResult.to_excel('optimization_result.xlsx')

##画图
Y= [i for i in function1_value_list]##环境效益
X= [k for k in function2_value_list]##成本
plt.xlabel('cost', fontsize=15)
plt.ylabel('environment', fontsize=15)
plt.scatter(X, Y)
plt.show()
