
# 读文件
with open(r"C:\Users\Hangyu\Desktop\gt_img_902.txt",encoding='utf-8-sig') as f:
    f.readlines()
    List = []
    for item in f:
        component = item.split(',')
        component[0],component[1] = int(component[0])/2,int(int(component[1])/2+140)
        component[2],component[3] = int(component[2])/2,int(int(component[3])/2+140)
        component[4],component[5] = int(component[4])/2,int(int(component[5])/2+140)
        component[6],component[7] = int(component[6])/2,int(int(component[7])/2+140)
        List.append(component)

# 写文件
with open(r"C:\Users\Hangyu\Desktop\gt_img_902_new.txt",'w',encoding='UTF-8-sig')as f:
    for component in List:
        temp = ','.join(str(i) for i in component)
        f.write(temp)


