import random
import pickle

'''原始算法'''

def randomPoint(min, max):
	num = random.random() * max + 1
	if num < min:
		num += min
	elif num > max:
		num = max
	return num


def individual(xMin, xMax, yMin, yMax):
	return [randomPoint(xMin, xMax), randomPoint(yMin, yMax)]


def population(popSize, xMin, xMax, yMin, yMax):
	return [individual(xMin, xMax, yMin, yMax) for _ in range(popSize)]


def fitness(indi):
	x1 = indi[0]
	x2 = indi[1]
	fit = 100 * (x1**2 + x2**2) + (1 - x1**2)**2
	return fit


def grade(pop):
	"""
    种群的平均适应度值
    """
	summed = 0
	for i in range(len(pop)):
		summed += fitness(pop[i])

	return summed / (len(pop) * 1.0)


def evolve(pop, xMin, xMax, yMin, yMax, retain=0.00, select=0.02, crossover=0.6, mutate=0.01):
	"""
	种群进化
    pop:                Population.
    xMin/Max, yMin/Max: 变量边界.
    retain:             精英个体保留比例.
    select:             随机保留父代2%的个体进入子代(为了保证种群多样性，最小为2%).
    mutate:             变异概率.
    """
	# Grade the population and store tuples of individuals and their fitnesses.
	# Then sort the graded population based on fitness, higher fitness at the
	# front, and only save the sorted individuals, exclude thier fitnesses.
	graded = [(fitness(x), x) for x in pop]
	graded = [x[1] for x in sorted(graded, reverse=True)]
	retain_length = int(len(graded) * retain)
	parents = graded[:2]  # Only two parents with best fitness.
	retained = graded[2:retain_length + 2]  # Retain 20%, except for parents.

	# Randomly add other individuals to promot genetic diversity.
	for indi in graded[retain_length + 2:]:
		if select > random.random():
			retained.append(indi)

	# 变异
	for indi in graded:
		if mutate > random.random():
			positionToMutate = random.randint(0, len(indi) - 1)
			# 对 x1 进行变异
			if positionToMutate == 0:
				indi[positionToMutate] = randomPoint(xMin, xMax)
			# 对 x2 进行变异
			else:
				indi[positionToMutate] = randomPoint(yMin, yMax)

	# 两点交叉
	retainedLength = len(retained)
	# Basically the length of the remaining population that needs to
	# be made to complete the new population.
	desiredLength = len(pop) - len(parents) - retainedLength
	children = []

	while len(children) < desiredLength:
		# 满足交叉概率
		if crossover > random.random():
			p1 = retained[random.randint(0, retainedLength - 1)]
			randParent = random.randint(0, len(parents) - 1)
			p2 = parents[randParent]
			half = int(len(p1) / 2)
			if randParent == 0:
				child = p1[:half] + p2[half:]
			else:
				child = p2[:half] + p1[half:]
			children.append(child)
		# 满足变异概率
			children.append(child)

	parents.extend(retained)
	parents.extend(children)
	return parents


# *************************************************************************
maxGen = 50
generation = 0
genFitness = 0
bestX = 0
bestY = 0
bestFitness = 0  # Only the best (highest) fitness is stored here.
popSize = 300

xMin = -100 # x1的下界
xMax = 100  # x1的上届
yMin = -100 # x2的下界
yMax = 100  # x2的上届

# Initialize population.
pop = population(popSize, xMin, xMax, yMin, yMax)

# Grade generation 0 (get avg fitness).
genFitness = grade(pop)
bestFitness = genFitness

# Keep begin fitness history!
fitnessHistory = [genFitness]

# While best fitness hasn't been changed in 20 generations.
while generation < maxGen:
	generation += 1
	pop = evolve(pop, xMin, xMax, yMin, yMax)

	# Grade the generation.
	genFitness = grade(pop)

	# 保存最好解
	if genFitness > bestFitness:
		bestFitness = genFitness
	fitnessHistory.append(genFitness)

# 结果输出
bestX = pop[0][0]
bestY = pop[0][1]
print("\n****************************************")
print("***** 迭代次数:     ", generation)
print("****************************************\n")
print("Best X and Y: ", bestX, " ", bestY)
print("Best fitness: ", fitnessHistory[len(fitnessHistory) - 1])
print("\n\n")

# 保存结果
with open('data_ga1.pkl','wb') as f:
	pickle.dump(fitnessHistory, f)


