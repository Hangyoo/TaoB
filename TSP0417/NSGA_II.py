import numpy as np
from TSP0417.Individual import *
from TSP0417.TSP import *
from TSP0417.Population import *
import TSP0417.GA as ga
from TSP0417.NSGA_II import *
from TSP0417.Draw import *
import warnings
import pickle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class NSGA_II:

    # evolution number
    num = 300
    # population
    pop = []
    population = None

    def __init__(self):
        self.disList = []
        self.costList = []
        self.dis_record = 1e10
        self.cost_record = 1e10

    def fast_nondominated_sort(self, P):
        f = []
        f1 = self.f1_dominate(P)
        if len(f1) != 0:
            f.append(f1)
        i = 0
        while len(f1) != 0:
            temp = []
            for p in f1:
                for q in p.Sp:
                    q.Np = q.Np - 1
                    if q.Np == 0:
                        q.rank = i + 1
                        temp.append(q)
            i = i + 1
            f1 = temp
            if len(f1) != 0:
                f.append(f1)

        return f

    def f1_dominate(self, P):
        f1 = []
        for i in P:
            for j in P:
                if i != j:
                    if self.is_dominate(i, j):
                        if j not in i.Sp:
                            i.Sp.append(j)
                    elif self.is_dominate(j, i):
                        i.Np = i.Np + 1

            # i belongs to the first front
            if i.Np == 0:
                i.rank = 0
                f1.append(i)

        return f1

    def is_dominate(self, x, y):
        x_cost = x.cost
        y_cost = y.cost
        x_distance = x.distance
        y_distance = y.distance

        if x_cost < y_cost:
            # if x dominates y
            if x_distance < y_distance:
                return True
            else:
                return False

        else:
            return False

    def sort_cost(self, Fi):
        length = len(Fi)

        for i in range(1, length):
            key = Fi[i]
            j = i - 1

            while j >= 0:

                if Fi[j].cost > key.cost:
                    Fi[j + 1] = Fi[j]
                    Fi[j] = key
                j = j - 1

        return Fi

    def sort_distance(self, Fi):
        length = len(Fi)

        for i in range(1, length):
            key = Fi[i]
            j = i - 1

            while j >= 0:

                if Fi[j].distance > key.distance:
                    Fi[j + 1] = Fi[j]
                    Fi[j] = key
                j = j - 1

        return Fi

    def sort_crowding(self, Fi):
        length = len(Fi)

        for i in range(1, length):
            key = Fi[i]
            j = i - 1

            while j >= 0:

                if Fi[j].crowding_distance < key.crowding_distance:
                    Fi[j + 1] = Fi[j]
                    Fi[j] = key
                j = j - 1

        return Fi

    def crowding_distance(self, Fi):
        num = len(Fi)
        cost_max = Fi[0].cost
        cost_min = Fi[0].cost
        distance_max = Fi[0].distance
        distance_min = Fi[0].distance

        for i in Fi:
            i.crowding_distance = 0

        for i in Fi:
            if i.cost > cost_max:
                cost_max = i.cost
            if i.cost < cost_min:
                cost_min = i.cost
            if i.distance > distance_max:
                distance_max = i.distance
            if i.distance < distance_min:
                distance_min = i.distance

        Fi[0].crowding_distance = float('inf')
        Fi[-1].crowding_distance = float('inf')
        Fi = self.sort_cost(Fi)
        for f in range(1, num - 1):
            a = Fi[f + 1].cost - Fi[f - 1].cost
            b = cost_max-cost_min
            Fi[f].crowding_distance =  Fi[f].crowding_distance + a / b

        Fi = self.sort_distance(Fi)
        for f in range(1, num - 1):
            a = Fi[f + 1].distance - Fi[f - 1].distance
            b = distance_max - distance_min
            Fi[f].crowding_distance = Fi[f].crowding_distance + a / b

    def C_indicator(self, A, B):
        num = 0
        for i in range(len(B)):
            count = 0
            for j in range(len(A)):
                if(self.is_dominate(A[j], B[i])):
                    count = count + 1
            if(count != 0):
                num = num + 1
        C_AB = float(num/len(B))

        return C_AB

    def run(self, tsp):
        pop = Population()
        self.population = pop.create_pop(tsp)

        front = self.fast_nondominated_sort(self.population)
        for f in front:
            self.crowding_distance(f)

        self.population = []
        for f in front:
            self.population.extend(f)
        next = pop.next_pop(self.population, tsp)

        result = []
        print(f'NSGAII算法正在执行，共需迭代{self.num}次')
        for i in range(self.num):
            if i % 100 == 0:
                print(f'剩余迭代次数：{self.num - i}')
            self.population.extend(next)
            front = self.fast_nondominated_sort(self.population)
            new = []
            num = 0
            while (len(new) + len(front[num])) <= pop.size:
                self.crowding_distance(front[num])
                new.extend(front[num])
                num = num + 1

            front[num] = self.sort_crowding(front[num])
            new.extend(front[num][0:(pop.size - len(new))])
            result = front[0]

            # 记录距离的最好值
            if (result[0].distance < self.dis_record):
                self.dis_record = result[0].distance
                self.disList.append(self.dis_record)
            else:
                self.disList.append(self.dis_record)

            # 记录成本的最好值
            if (result[0].cost < self.cost_record):
                self.cost_record = result[0].cost
                self.costList.append(self.cost_record)
            else:
                self.costList.append(self.cost_record)

            self.population = new
            next = pop.next_pop(self.population, tsp)
        # 保存文件
        print('运行数据保存到.pkl中, 运行NSGA2和NSGA2_improve后可进行对比.')
        with open('distance_nsga2.pkl','wb') as f:
            pickle.dump(self.disList,f)
        with open('cost_nsga2.pkl','wb') as f:
            pickle.dump(self.costList,f)
        return result, self.disList, self.costList

    def draw_all(self, result, tsp):
        dw = Draw()
        dw.bound_x = [np.min(tsp.cities[:, 0]), np.max(tsp.cities[:, 0])]
        dw.bound_y = [np.min(tsp.cities[:, 1]), np.max(tsp.cities[:, 1])]
        dw.set_xybd(dw.bound_x, dw.bound_y)
        dw.draw_points(tsp.cities[:, 0], tsp.cities[:, 1])
        n = len(result.indiv)
        for i in range(n):
            city = result.indiv[i]
            dw.draw_text(tsp.cities[city][0], tsp.cities[city][1], tsp.city_name[city], 8)

        for i in range(n - 1):
            city_from = result.indiv[i]
            city_to = result.indiv[i + 1]
            dw.draw_line(tsp.cities[city_from], tsp.cities[city_to])

        start = result.indiv[0]
        end = result.indiv[-1]
        dw.draw_line(tsp.cities[end], tsp.cities[start])

        dw.plt.show()

# 绘制目标函数迭代图
def plot_chart(result,disList,costList):
    # 绘图设置(显示中文)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    X,Y = [],[]
    for item in result:
        X.append(item.cost)
        Y.append(item.distance)
    plt.scatter(X,Y)
    plt.xlabel('Totalcost')
    plt.ylabel('Distance')
    plt.title("改进前Pareto Front")
    plt.show()

    # 距离迭代图
    plt.plot(disList,'end')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title("改进前距离收敛图")
    plt.show()
    # 成本迭代图
    plt.plot(costList, 'end')
    plt.xlabel('Iteration')
    plt.ylabel('Totalcost')
    plt.title("改进前成本收敛图")
    plt.show()



def main():
    nsga = NSGA_II()
    tsp = TSP()
    result,disList,costList = nsga.run(tsp)
    print(f'result:{result}')
    print(f'distance:{result[0].distance}')
    print(f'cost:{result[0].cost}')
    nsga.draw_all(result[0], tsp)
    plot_chart(result,disList, costList)


if __name__ == '__main__':
    main()









