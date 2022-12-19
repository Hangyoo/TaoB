import numpy as np

# 将获得的所有最优解写入文件
def print_object_to_file(path,PF):
    if type(PF) is not list:
        PF = [PF]

    with open(path, 'w') as of:
        for point in PF:
            for function_value in point:
                of.write(str(function_value) + ' ')
            of.write('\n')

def Non_donminated_sorting(chroms_obj_record):
    # 非支配排序
    length = len(chroms_obj_record)
    f = np.reshape(chroms_obj_record,(length,len(chroms_obj_record[0])))
    Rank = np.zeros(length)  # [0. 2. 1. 1. 1. 0. 0. 0. 2. 1.]
    front = []     # [[0, 5, 6, 7], [2, 3, 4, 9], [1, 8]]
    rank = 0

    n_p = np.zeros(length)
    s_p = []
    for p in range(length):
        a = (f[p, :] - f[:, :] <= 0).all(axis=1)
        b = (~((f[p, :] - f[:, :] == 0).all(axis=1)))
        loc = np.where(a & b)[0].tolist()
        s_p.append(loc)
        temp1 = np.where(((f[p, :] - f[:, :] >= 0).all(axis=1)) & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0]
        n_p[p] = len(temp1)  # p所支配个数
    # 添加第一前沿
    front.append(list(np.where(n_p == 0)[0]))

    while len(front[rank]) != 0:    # 生成其他前沿
        elementset = front[rank]
        n_p[elementset] = float('inf')
        Rank[elementset] = rank
        rank += 1

        for i in elementset:
            temp = s_p[i]
            n_p[temp] -= 1
        front.append(list(np.where(n_p == 0)[0]))
    front.pop()
    # 第一前沿
    parero_solution_obj = []
    first = front[0]
    for index in first:
        parero_solution_obj.append(chroms_obj_record[index])
    return parero_solution_obj

def algorithm_pareto_merge():
    # 将不同算法获得的PF进行合并 并写入文件
    instances = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    for ins in instances:
        reference_front_path1 = r"C:\Users\Hangyu\Desktop\TVCS\CRA\front\TI"+ins+".pf"    # MOEA/D-CRA
        reference_front_path2 = r"C:\Users\Hangyu\Desktop\TVCS\DRA\front\TI"+ins+".pf"    # MOEA/D-DRA
        reference_front_path3 = r"C:\Users\Hangyu\Desktop\TVCS\M2M\front\TI"+ins+".pf"    # MOEA/D-M2M
        # reference_front_path4 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA2\front\TI"+ins+".pf"  # NSGA2
        reference_front_path5 = r"C:\Users\Hangyu\Desktop\TVCS\NSGA3\front\TI"+ins+".pf"  # NSGA3
        reference_front_path6 = r"C:\Users\Hangyu\Desktop\TVCS\PD\front\TI"+ins+".pf"     # MOEA/D-PD
        reference_front_path7 = r"C:\Users\Hangyu\Desktop\TVCS\SVM\front\TI"+ins+".pf"    # MOEA/D-SVM
        # reference_front_path8 = r"C:\Users\Hangyu\Desktop\TVCS\PD_NL\front\TI"+ins+".pf"  # MOEA/D-PD-NL
        # reference_front_path9 = r"C:\Users\Hangyu\Desktop\TVCS\PD_NP\front\TI"+ins+".pf" # MOEA/D-PD-NP

        reference_front1 = np.loadtxt(reference_front_path1).tolist()
        reference_front2 = np.loadtxt(reference_front_path2).tolist()
        reference_front3 = np.loadtxt(reference_front_path3).tolist()
        # reference_front4 = np.loadtxt(reference_front_path4).tolist()
        reference_front5 = np.loadtxt(reference_front_path5).tolist()
        reference_front6 = np.loadtxt(reference_front_path6).tolist()
        reference_front7 = np.loadtxt(reference_front_path7).tolist()
        # reference_front8 = np.loadtxt(reference_front_path8).tolist()
        # reference_front9 = np.loadtxt(reference_front_path9).tolist()

        all = reference_front1 + reference_front2 + reference_front3 + reference_front5 \
              + reference_front6 + reference_front7 #+ reference_front8 + reference_front9

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = r"C:\Users\Hangyu\Desktop\TVCS\PF\TI"+ins+".pf"
        print_object_to_file(reference_front_path, now_reference_front)
        print(f"完成instance TI-{ins} 前沿合并！")

def file_pareto_merge():
    # 将每个算法获得的PF进行合并 并写入文件 (调参时使用)
    instances = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15']
    independentRuns = 20 # 算法独立运行的次数
    print(" 0: CRA\n","1: DRA\n","2: M2M\n","3: NSGA2\n","4: NSGA3\n","5: SVM\n","6: MOEA/D-PD\n","7: MOEA/D-PD-NP\n","8: MOEA/D-PD-NL\n")
    idx = int(input("请输入要合并的算法序号:"))
    algorithm = ["CRA","DRA","M2M","NSGA2","NSGA3","SVM","PD","NP","NL"]

    # 判断合并哪种算法的pareto
    if algorithm[idx] == "CRA":
        path = r"C:\Users\Hangyu\Desktop\TVCS\CRA\FUN.MOEAD-CRA.Cloudcpt"  # MOEA/D-CRA
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\CRA\front"  # MOEA/D-CRA
    elif algorithm[idx] == "DRA":
        path = r"C:\Users\Hangyu\Desktop\TVCS\DRA\FUN.MOEAD-DRA.Cloudcpt"  # MOEA/D-DRA
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\DRA\front"  # MOEA/D-DRA
    elif algorithm[idx] == "M2M":
        path = r"C:\Users\Hangyu\Desktop\TVCS\M2M\FUN_MOEAD-M2M.Cloudcpt"  # MOEA/D-M2M
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\M2M\front"  # MOEA/D-M2M
    elif algorithm[idx] == "NSGA2":
        path = r"C:\Users\Hangyu\Desktop\TVCS\NSGA2\FUN.NSGAII.Cloudcpt"   # NSGA2
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\NSGA2\front"   # NSGA2
    elif algorithm[idx] == "NSGA3":
        path = r"C:\Users\Hangyu\Desktop\TVCS\NSGA3\FUN.NSGA3.Cloudcpt"    # NSGA3
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\NSGA3\front"    # NSGA3
    elif algorithm[idx] == "SVM":
        path = r"C:\Users\Hangyu\Desktop\TVCS\SVM\FUN.MOEAD-SVM.Cloudcpt"  # MOEA/D-SVM
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\SVM\front"  # MOEA/D-SVM
    elif algorithm[idx] == "PD":
        path = r"C:\Users\Hangyu\Desktop\TVCS\PD\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-PD
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\PD\front"    # MOEA/D-PD
    elif algorithm[idx] == "NP":
        path = r"C:\Users\Hangyu\Desktop\TVCS\PD_NP\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-NP
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\NP\front"    # MOEA/D-NP
    elif algorithm[idx] == "NL":
        path = r"C:\Users\Hangyu\Desktop\TVCS\PD_NL\FUN.MOEAD-PD.Cloudcpt"    # MOEA/D-NL
        path_pront = r"C:\Users\Hangyu\Desktop\TVCS\NL\front"    # MOEA/D-NL
    else:
        print('算法名称输入有误')

    for i in range(15):
        ins = instances[i]
        all = []
        for j in range(independentRuns):
            reference_front_path = path + str(i) + "_" + str(j) + ".txt"
            reference_front = np.loadtxt(reference_front_path).tolist()
            all += reference_front
        print(f"完成instance TI-{i+1} 前沿合并！")

        now_reference_front = Non_donminated_sorting(all)

        reference_front_path = path_pront + r"\TI"+ins+".pf"
        print_object_to_file(reference_front_path, now_reference_front)

if __name__ == "__main__":
    file_pareto_merge()
    # algorithm_pareto_merge()