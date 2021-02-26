import numpy as np
import pandas as pd
import ga
import pickle
import Fitness
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

#数据预处理
data = pd.read_excel(r"C:\Users\Hangyu\Desktop\混凝土抗压强度数据整理.xlsx")

data_inputs = np.array(data.iloc[:,0:8])
data_outputs = np.array(data.iloc[:,8]).reshape(-1,1)

# 数据标准化
mm = MinMaxScaler()
labels = mm.fit_transform(data_outputs)
#data_inputs = mm.fit_transform(data_inputs)

# 划分测试集和验证集
# x_train, x_test, y_train, y_test = train_test_split(data_inputs,labels,random_state=24)
data_inputs, x_test, data_outputs, y_test = train_test_split(data_inputs,labels,random_state=24)

"""
Genetic algorithm parameters:
    Mating Pool Size (Number of Parents)
    Population Size
    Number of Generations
    Mutation Percent
"""

sol_per_pop = 8
num_parents_mating = 2
num_generations = 200
mutation_percent = 10

# 创建初始种群
initial_pop_weights = []
for curr_sol in np.arange(0, sol_per_pop):
    HL1_neurons = 200
    input_HL1_weights = np.random.uniform(low=-1, high=1,
                                             size=(data_inputs.shape[1], HL1_neurons))
    HL2_neurons = 100
    HL1_HL2_weights = np.random.uniform(low=-1, high=1,
                                             size=(HL1_neurons, HL2_neurons))
    output_neurons = 1
    HL2_output_weights = np.random.uniform(low=-1, high=1,
                                              size=(HL2_neurons, output_neurons))

    initial_pop_weights.append(np.array([input_HL1_weights, 
                                                HL1_HL2_weights, 
                                                HL2_output_weights]))

pop_weights_mat = np.array(initial_pop_weights)
pop_weights_vector = ga.mat_to_vector(pop_weights_mat)

best_outputs = []
accuracies = np.empty(shape=(num_generations))

for generation in range(num_generations):
    print("Generation : ", generation)

    # converting the solutions from being vectors to matrices.
    pop_weights_mat = ga.vector_to_mat(pop_weights_vector, 
                                       pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = Fitness.fitness(pop_weights_mat,
                          data_inputs, 
                          data_outputs, 
                          activation="sigmoid")
    accuracies[generation] = min(fitness)


    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(pop_weights_vector, 
                                    fitness.copy(), 
                                    num_parents_mating)


    # Generating next generation using crossover.
    offspring_crossover = ga.crossover(parents,
                                       offspring_size=(pop_weights_vector.shape[0]-parents.shape[0], pop_weights_vector.shape[1]))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = ga.mutation(offspring_crossover, 
                                     mutation_percent=mutation_percent)


    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

accuracies = sorted(map(lambda x:-x,accuracies))
pop_weights_mat = ga.vector_to_mat(pop_weights_vector, pop_weights_mat)
# 绘制MSE迭代曲线
plt.plot(accuracies, linewidth=5, color="black")
plt.title("R2 value with Iteration")
plt.xlabel("Iteration", fontsize=13)
plt.ylabel("MSE", fontsize=13)
plt.xticks(np.arange(0, num_generations+1,50), fontsize=13)
plt.show()
# 储存
f = open("weights_"+str(num_generations)+"_iterations_"+str(mutation_percent)+"%_mutation.pkl", "wb")
pickle.dump(pop_weights_mat, f)
f.close()
