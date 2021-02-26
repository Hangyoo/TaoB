import random
import shutil
import copy

class Readtext():
    def __init__(self,path):
        self.path = path
        self.machinesNb = 0
        self.jobsnNb = 0
        self.jobs = []
        self.info = {}
        self.each_job_oper_num = []

    # def Read_text(self):
    #     # 读取原文件
    #     file = open(self.path, 'r')
    #     # 创建模糊加工时间文件
    #     shutil.copy(self.path, self.path + "_fuzzy")
    #     fileFuzzy = open(self.path + "_fuzzy", 'w')
    #
    #     firstLine = file.readline()
    #     firstLineValues = list(map(int, firstLine.split()[0:2]))
    #     self.jobsnNb = firstLineValues[0]      #获取工件数
    #     self.machinesNb = firstLineValues[1]  #获取机器数
    #     self.each_job_oper_num = [[] for _ in range(self.jobsnNb)]
    #
    #     for i in range(self.jobsnNb):
    #         currentLine = file.readline()     #读取每一行
    #         currentLineValues = list(currentLine.split())
    #         fuzzyLineValues = copy.copy(currentLineValues)
    #
    #         #currentLineValues = [7, 1, 6, 1, 2, 6, 1, 4, 7, 1, 1, 6, 2, 6, 7, 3, 1, 3, 2, 3, 4, 8, 3, 2, 1, 6, 2, 1, 7, 2]
    #         job = []
    #         j = 1
    #         while j < len(currentLineValues):
    #             operations = []                     #每个工序加工列表
    #             path_num = int(currentLineValues[j])     #获取本工序的加工路线个数
    #             j = j+1
    #             for path in range(path_num):
    #                 machine = int(currentLineValues[j])  #获取加工机器
    #                 j = j+1
    #                 processingTime = int(currentLineValues[j])    #获取加工时间
    #                 # # 模糊时间确定
    #                 low = random.randint(1, processingTime-1) if processingTime>1 else 1
    #                 up = random.randint(processingTime+1,2*processingTime)
    #                 fuzzyTime = [low, processingTime, up]
    #                 fuzzyLineValues[j] = fuzzyTime
    #                 j = j+1
    #                 operations.append({'machine': machine, 'processingTime': processingTime})
    #             job.append(operations)
    #         self.jobs.append(job)   # [[【工序1】，【】，【】，【】，],[工件2],[]]
    #
    #         # 写入模糊加工时间文件
    #         for i in fuzzyLineValues:
    #             if type(i) is list:
    #                 fileFuzzy.write(str(i[0])+" ")
    #                 fileFuzzy.write(str(i[1])+" ")
    #                 fileFuzzy.write(str(i[2])+" ")
    #             else:
    #                 fileFuzzy.write(str(i) + " ")
    #         fileFuzzy.write("\n")
    #     fileFuzzy.close()
    #     # 关闭原文件
    #     file.close()
    #     self.info = {'machinesNb': self.machinesNb, 'jobs': self.jobs, 'jobsnum': self.jobsnNb}
    #     return self.info

    def readtext(self):
        # 读取原文件
        file = open(self.path, 'r')
        # 创建模糊加工时间文件

        firstLine = file.readline()
        firstLineValues = list(map(int, firstLine.split()[0:2]))
        self.jobsnNb = firstLineValues[0]      #获取工件数
        self.machinesNb = firstLineValues[1]  #获取机器数
        self.each_job_oper_num = [[] for _ in range(self.jobsnNb)]

        for i in range(self.jobsnNb):
            currentLine = file.readline()     #读取每一行
            currentLineValues = list(currentLine.split())

            job = []
            j = 1
            while j < len(currentLineValues):
                operations = []                     #每个工序加工列表
                path_num = int(currentLineValues[j])     #获取本工序的加工路线个数
                j = j+1
                for path in range(path_num):
                    machine = int(currentLineValues[j])  #获取加工机器
                    j = j+2
                    processingTime = int(currentLineValues[j])    #获取加工时间
                    processingTimeFuzzy = (int(currentLineValues[j-1]),int(currentLineValues[j]),int(currentLineValues[j+1]))
                    j = j+2
                    operations.append({'machine': machine, 'processingTime': processingTime,'processingTimeFuzzy':processingTimeFuzzy})
                job.append(operations)
            self.jobs.append(job)   # [[【工序1】，【】，【】，【】，],[工件2],[]]

        self.info = {'machinesNb': self.machinesNb, 'jobs': self.jobs, 'jobsnum': self.jobsnNb}
        return self.info

if __name__ ==  '__main__':
    patch = r'C:\Users\Hangyu\PycharmProjects\fjspAndConverttime\taobao\Benchmark\Mk10.fjs'
    a = Readtext(patch)
    print(a.readtext())
