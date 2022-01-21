__author__ = 'lly'
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import re

class NODE:
    def __init__(self, name, dem, x, y):
        self.name = name
        self.dem = dem
        self.xcor = x
        self.ycor = y
        self.invert = None # invert EI.
        self.depth = None # Max. Depth
        self.leaves = []
        self.root = None
        self.downpipe = None
        self.subc = []
        self.subcQ = 0
        self.t2 = 0 # t = t1 + m * t2
node_set = {}
node_info = pd.read_excel('node.xlsx')
N_node = node_info.shape[0]#node数量
for i in range(N_node):
    node_set[node_info.node_name[i]] = NODE(node_info.node_name[i], node_info.dem[i], node_info.x_cor[i], node_info.y_cor[i])
print(node_set)