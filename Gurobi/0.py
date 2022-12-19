from gurobipy import *

# 导入模型
m = Model("case 1")
# 添加变量x1
x1 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x1")
# 添加变量x2
x2 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x2")
# 添加变量x3
x3 = m.addVar(lb=0.0,vtype=GRB.INTEGER,name="x3")

# 设定目标函数
m.setObjective(2*x1+3*x2+4*x3,GRB.MAXIMIZE)

# 设定约束条件1
m.addConstr(1.5*x1+3*x2+5*x3<=100,"c1")

# 设定约束条件2
m.addConstr(280*x1+250*x2+400*x3<=60000,"c1")

# 优化
m.optimize()

print('最优目标函数值：', m.objVal)