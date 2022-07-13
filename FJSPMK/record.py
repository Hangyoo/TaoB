

def record(chroms_obj_record,convergence,before_value):
    # 记录每代最小完工时间
    f1 = min([item[0] for item in chroms_obj_record.values()])
    if f1 < before_value[0]:
        before_value[0] = f1
    convergence["makespan"].append(min(f1,before_value[0]))

    # 记录每代最小成本
    f2 = min([item[1] for item in chroms_obj_record.values()])
    if f2 < before_value[1]:
        before_value[1] = f2
    convergence["cost"].append(min(f2,before_value[1]))

    return convergence,before_value