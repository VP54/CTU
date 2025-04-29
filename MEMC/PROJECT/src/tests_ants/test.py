from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matrix_funcs import GetFoodMatrix, CreatePheromoneScores

def simulate(num_ants, NUM_STEPS, all_ants):
    lst_values = []
    food_matrix_slicer, p = GetFoodMatrix(), CreatePheromoneScores()
    matrix = np.random.rand(NUM_STEPS, NUM_STEPS)
    for ant in range(num_ants):
        matrix = matrix * np.random.uniform(1, 2)
        counter = 0
        lst_values = []
        for step in range(NUM_STEPS):
            step_value, prob_stop = p.compute_probabilities(matrix, step, food_matrix_slicer)
            step_value += step_value
            if counter == 0:
                counter += 1
                step_value = (NUM_STEPS) / 2
            if prob_stop == 1:
                return all_ants
            lst_values.append(step_value)
        all_ants.append(lst_values)
    return all_ants

NUM_STEPS = 1000
num_ants, prob_left, x, step_value, x_axis = 10, .2, 0, 0, np.linspace(1, NUM_STEPS, NUM_STEPS)
a, b = 0, 1 # uniform
mu, sigma = 0, .5 # Normal
all_ants, lst_values = [], []
x_axis = np.linspace(1, NUM_STEPS, NUM_STEPS)
plt.figure(figsize=(10,12))
DISTRIBUTION = "uniform"

ants = simulate(num_ants, NUM_STEPS, all_ants)
print(ants)

for i in range(len(all_ants)):
    plt.scatter(ants[i], x_axis, s=3)