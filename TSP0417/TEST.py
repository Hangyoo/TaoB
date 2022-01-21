from TSP0417.Individual import *
from TSP0417.TSP import *
from TSP0417.Population import *
import TSP0417.GA as ga
from TSP0417.NSGA_II import *
from TSP0417.Draw import *
import time

nsga = NSGA_II()
tsp = TSP()
count = 0
for i in range(10):
    result = nsga.run(tsp)
    print('result')
    print(result)
    print('distance')
    print(result[0].distance)
    print('cost')
    print(result[0].cost)
    count += result[0].distance

print(count/10)

#nsga.draw_all(result[0], tsp)