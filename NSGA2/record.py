

def record(chroms_obj_record,convergence,before_value):
    # 记录每代最小TN
    f1 = min([item[0] for item in chroms_obj_record.values()])
    if f1 < before_value[0]:
        before_value[0] = f1
    convergence["TN"].append(min(f1,before_value[0]))

    # 记录每代最小TP
    f2 = min([item[1] for item in chroms_obj_record.values()])
    if f2 < before_value[1]:
        before_value[1] = f2
    convergence["TP"].append(min(f2,before_value[1]))

    # 记录每代最小COST
    f3 = min([item[2] for item in chroms_obj_record.values()])
    if f3 < before_value[2]:
        before_value[2] = f3
    convergence["COST"].append(min(f3,before_value[2]))

    return convergence,before_value