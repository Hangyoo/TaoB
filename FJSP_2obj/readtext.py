class Readtext():
    def __init__(self,path):
        self.path = path
        self.machinesNb = 0
        self.jobsnNb = 0
        self.jobs = []
        self.info = {}
        self.each_job_oper_num = []

    def readtext(self):
        file = open(self.path, 'r')
        firstLine = file.readline()
        firstLineValues = list(map(int, firstLine.split()[0:2]))
        self.jobsnNb = firstLineValues[0]      #获取工件数
        self.machinesNb = firstLineValues[1]  #获取机器数
        self.each_job_oper_num = [[] for _ in range(self.jobsnNb)]

        for i in range(self.jobsnNb):
            currentLine = file.readline()     #读取每一行
            currentLineValues = list(map(float, currentLine.split()))
            #currentLineValues = [7, 1, 6, 1, 2, 6, 1, 4, 7, 1, 1, 6, 2, 6, 7, 3, 1, 3, 2, 3, 4, 8, 3, 2, 1, 6, 2, 1, 7, 2]
            job = []
            j = 1
            while j < len(currentLineValues):
                operations = []                     #每个工序加工列表
                path_num = int(currentLineValues[j])     #获取本工序的加工路线个数
                # print(1111,path_num)
                j = j+1
                for path in range(path_num):
                    machine = currentLineValues[j]  #获取加工机器
                    # print(2222,currentLineValues[j])
                    j = j+1
                    processingTime = currentLineValues[j]    #获取加工时间
                    # print(3333,currentLineValues[j])
                    j = j+1
                    # print(j)
                    operations.append({'machine': machine, 'processingTime': processingTime})
                job.append(operations)
            self.jobs.append(job)   # [[【工序1】，【】，【】，【】，],[工件2],[]]
        file.close()
        self.info = {'machinesNb': self.machinesNb, 'jobs': self.jobs, 'jobsnum': self.jobsnNb}
        return self.info

if __name__ ==  '__main__':
    patch = r'C:\Users\Hangyu\Desktop\JmetalTB\FJSP_2obj\realworld.fjs'
    a = Readtext(patch)
    print(a.readtext())
