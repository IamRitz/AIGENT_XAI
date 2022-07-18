import gurobipy as gp
from gurobipy import GRB

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()
m = gp.Model("Model", env=env)
m.setParam('NonConvex', 2)
x = 1
limits = []
input_vars = []
changes = []
val_max = 100
epsilon_max = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name="epsilon_max")
i = 0

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v12 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v12")
m.addConstr(v12==(changes[i] + 1)*x)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v22 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v22")
m.addConstr(v22==(changes[i] + 1)*x)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v13 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v13")
m.addConstr(v13==(changes[i] + 0.01)*v12)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v23 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v23")
m.addConstr(v23==(changes[i] + 100)*v22)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v14 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v14")
m.addConstr(v14==(changes[i] + 1000)*v13)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
v24 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="v24")
m.addConstr(v24==(changes[i] + 0.01)*v23)
m.update()
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
i += 1
limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
o1 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="o1")
m.addConstr(o1==(changes[i-1] + 1)*v14 + (changes[i] + 1)*v24)
i += 1

limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
i += 1
limits.append(m.addVar(vtype=GRB.BINARY))
changes.append(m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS))
m.addConstr(changes[i]-epsilon_max*limits[i]<=0)
m.addConstr(changes[i]+epsilon_max*limits[i]>=0)
o2 = m.addVar(lb = -val_max, ub = val_max, vtype=GRB.CONTINUOUS, name ="o2")
m.addConstr(o2==(changes[i-1] -1)*v14 + (changes[i] -1)*v24)
i += 1

m.addConstr(o2-o1>=0.001)
sumX = gp.quicksum(limits)
m.addConstr(sumX==1)
m.update()
m.setObjectiveN(epsilon_max, index = 1, priority = 1)
# m.setObjectiveN(o1-o2, index = 0, priority = 1)
    
m.optimize()
if m.Status == GRB.INFEASIBLE:
    m.computeIIS()         
    m.write('model.ilp')
    print("Model infeasible.\n")
else:
    print("Model feasible.\n")

print("BigM\t Modification")
for i in range(len(changes)):
    print(limits[i].X, "\t", changes[i].X)
print("\nOutput1\t\t\t Output2")
print(o1.X, "\t", o2.X)
print()