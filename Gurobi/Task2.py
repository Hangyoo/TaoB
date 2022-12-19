from gurobipy import *

# 导入模型
m = Model("case 2")

P1 = 0.05
P2 = 0.07
P3 = 0.1
P4 = 0.1

# 添加变量x1
x1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x1")
x2 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x2")
x3 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x3")

# 添加变量x2
d1_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d1-")
d1_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d1+")
d2_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d2-")
d2_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d2+")
d3_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d3-")
d3_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d3+")
d4_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d4-")
d4_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d4+")
d5_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d5-")
d5_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d5+")
d6_0 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d6-")
d6_1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="d6+")


# 设定目标函数
m.setObjective(P1*d1_0 + P2*(d2_1+d3_1+d4_1) + P3*d5_0 + P4*d6_0, GRB.MINIMIZE)

# 设定约束条件1
m.addConstr(x1+x2+x3 <= 1000,"c1")
m.addConstr(x1+d1_0-d1_1 == 300,"c2")
m.addConstr(x1+d2_0-d2_1 <= 350,"c3")
m.addConstr(x2+d3_0-d3_1 <= 350,"c4")
m.addConstr(x3+d4_0-d4_1 <= 350,"c5")
m.addConstr(1000 - (x1+x2+x3) + d5_0 - d5_1 == 100,"c6")
m.addConstr(0.05*x1 + 0.07*x2 + 0.1*x3 + d6_0 - d6_1 == 100,"c7")

# 优化
m.optimize()

for i in range(len(m.getVars())):
    print('变量取值:',m.getVars()[i])
print('最优目标函数值:', m.objVal)