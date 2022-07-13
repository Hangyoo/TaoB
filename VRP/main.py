import math
from VRP.GA import GA
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
citys = []
f = pd.read_csv(r"city.txt",header=None,sep = ' ')
for i in range(f.shape[0]):
    a ,b = f.iloc[i,[1,2]]
    citys.append((a ,b))

class VRP(object):
      def __init__(self, aLifeCount = 100):
            self.citys = citys # 记录城市信息
            self.lifeCount = aLifeCount
            self.ga = GA(crossover= 0.8,
                         mutation= 0.1,
                         aLifeCount = self.lifeCount,
                         chrom_length= len(self.citys),
                         aMatchFun = self.matchFun())

            
      def distance(self, order):
            distance = 0.0
            for i in range(-1, len(self.citys) - 1):
                  a, b = order[i], order[i + 1]
                  city1, city2 = self.citys[a], self.citys[b]
                  # 计算两点之间距离
                  distance += math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)
            return distance


      def matchFun(self):
            return lambda x: 1.0 / self.distance(x.gene)


      def run(self, n):
            for i in range(n):
                  self.ga.next()
                  distance = self.distance(self.ga.best.gene)
                  print(f"当前迭代次数:{i},最短距离为：{round(distance,2)}")

      def get_best_idividual(self):
          print(self.ga.best.gene)
          return self.ga.best.gene

# 绘制路径
def chart(line,citys):
      X, Y = [], []
      for i in line:
          x,y = citys[i]
          X.append(x)
          Y.append(y)
      plt.scatter(X,Y)
      plt.plot(X,Y)
      plt.show()




def main():
      travel = VRP()
      # 这里设置迭代次数
      travel.run(500)
      # 获取最优方案
      best_individual = travel.get_best_idividual()
      # 绘制图片
      chart(best_individual,citys)



if __name__ == '__main__':
      main()


