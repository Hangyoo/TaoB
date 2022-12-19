from gurobipy import *

# 导入模型
m = Model("case 1")
# 添加变量x1
x11 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x11")
x12 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x12")
x13 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x13")
# 添加变量x2
x21 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x21")
x22 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x22")
x23 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x23")
# 添加变量x3
x31 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x31")
x32 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x32")
x33 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x33")

# 设定目标函数
m.setObjective(2*(x11+x12+x13)+3*(x21+x22+x23)+2.3*(x31+x32+x33)-1.7*(x11+x21+x31)-
               1.5*(x12+x22+x32)-1.2*(x13+x23+x33),GRB.MAXIMIZE)

# 设定约束条件1
m.addConstr(x11+x21+x31<=1500,"c1")
m.addConstr(x12+x22+x32<=1000,"c2")
m.addConstr(x13+x23+x33<=2000,"c3")
m.addConstr(0.15*(x11+x12+x13)<=x11,"c4")
m.addConstr(0.25*(x11+x12+x13)<=x12,"c5")
m.addConstr(0.20*(x21+x22+x23)<=x21,"c6")
m.addConstr(0.10*(x21+x22+x23)<=x22,"c7")
m.addConstr(0.25*(x31+x32+x33)==x31,"c8")
m.addConstr(0.40*(x31+x32+x33)>=x33,"c9")

# 优化
m.optimize()

for i in range(len(m.getVars())):
    print('变量取值:',m.getVars()[i])
print('最优目标函数值:', m.objVal)