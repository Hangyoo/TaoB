import numpy as np
import pandas as pd

machine = pd.read_excel('data.xlsx',sheet_name='加工')
time = pd.read_excel('data.xlsx',sheet_name='加工时间')
jobNum, oper = machine.shape


macnum = 0
OPR_NUM = [] # 统计工序数目
MAC_TIME = [] # 统计机器——加工时间
for job_idx in range(jobNum):
    opr_num = []
    mac_time = []
    for i in range(1,oper):
        if machine.iloc[job_idx,i] != 0:
            j = 0
            # print(type(machine.iloc[job_idx,i]))
            if issubclass(type(machine.iloc[job_idx,i]),str):
                # print(machine.iloc[job_idx, i])
                split_mac = machine.iloc[job_idx, i].split(',')
                split_time = time.iloc[job_idx, i].split(',')
                for k in range(len(split_mac)):
                    item = int(split_mac[k])
                    item_time = float(split_time[k])
                    if item > macnum:
                        macnum = item
                    mac_time.append((item,item_time))
                    # print(j,item)
                    j += 1
            else:
                if machine.iloc[job_idx, i] > macnum:
                    macnum = machine.iloc[job_idx, i]
                item = int(machine.iloc[job_idx, i])
                # print(time.iloc[job_idx, i])
                item_time = np.float(time.iloc[job_idx, i])
                j = 1
                mac_time.append((item, item_time))
            opr_num.append(j)
    OPR_NUM.append(opr_num)
    MAC_TIME.append(mac_time)
print(OPR_NUM)
print(MAC_TIME)

# 转换为文件:
filename = "realworld.fjs"

with open(filename,'w') as f:
    f.writelines([str(jobNum),str(' '),str(macnum)])
    f.write('\n')
    for i in range(jobNum):
        now = 0
        f.write(str(len(OPR_NUM[i])))
        f.write(' ')
        for j in range(len(OPR_NUM[i])):
            f.write(str(OPR_NUM[i][j]))
            f.write(' ')
            stop = now + OPR_NUM[i][j]
            while now < stop:
                f.write(str(MAC_TIME[i][now][0]) + ' ')
                f.write(str(MAC_TIME[i][now][1]) + ' ')
                print(now,now+OPR_NUM[i][j])
                now += 1

        f.write('\n')
