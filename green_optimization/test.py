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
def inp(a1,a2,a3,a4,b1,b2,c1,c2):
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
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_BR', '1', subc_object.outlet, subc_object.green*a1/ 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_VS', '1', subc_object.outlet, subc_object.green*a2 / 10000.0,\
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
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %d   %d\n" % (subc_name+'_roof_no', '1', subc_object.outlet, subc_object.roof*(1-a3) / 10000.0,\
                                                                                          100, width, 50, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %d   %d\n" % (subc_name+'_roof_with_GR', '1', subc_object.outlet, subc_object.roof*a3/ 10000.0,\
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
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_in_no_PP', '1', subc_object.outlet, subc_object.rdsq_in*(1-a4) / 10000.0,\
                                                                                          100, width, subc_object.slope, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_in_with_PP', '1', subc_object.outlet, subc_object.rdsq_in*a4/ 10000.0,\
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
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_no_PP', '1', subc_object.outlet, subc_object.rdsq_out*(1-a4)/ 10000.0,\
                                                                                          100, width, subc_object.slope, 0))
                subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_with_PP', '1',subc_object.outlet, subc_object.rdsq_out*a4 / 10000.0,\
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
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_BR', '1', subc_object.outlet, subc_object.green*b1 / 10000.0,\
                                                                                      0, width, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_green_with_VS', '1', subc_object.outlet, subc_object.green*b2 / 10000.0,\
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
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_no','1', subc_object.outlet, (subc_object.rdsq_in + subc_object.rdsq_out)*(1-c1-c2)/ 10000.0,\
                                                                                    100, math.sqrt((subc_object.rdsq_in + subc_object.rdsq_out)) * 9.59, subc_object.slope, 0))
            subcatchment_part.write("%s   %s   %s   %6.4f   %d   %6.3f   %5.3f   %d\n" % (subc_name+'_rdsq_out_with_PP', '1', subc_object.outlet, (subc_object.rdsq_in + subc_object.rdsq_out)*c1/ 10000.0,\
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

    with open ('pjxc_blank.inp') as f:
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


    with open ('pjxc.inp','w',encoding='utf-8') as f:
        for line in lines:
            f.write(line)

    return BR,VS,PP,GR

inp(0,0,0,0,0,0,0,0)