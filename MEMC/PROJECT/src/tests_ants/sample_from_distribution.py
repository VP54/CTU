from typing import Tuple
import numpy as np

def uniform_decision(a: float, b: float, prob_left:float, step_value: int):
    from_distribution = np.random.uniform(a, b)
    if from_distribution >= prob_left:
        step_value += 1
    else:
        step_value -= 1
    return step_value

def binomial_decision(prob_left: float, step_value: int):
    from_binomial = np.random.binomial(1, prob_left)
    if from_binomial[0] == 1:
        step_value += 1
    else:
        step_value -= 1
    return step_value

def normal_decision(mu: float, std: float, step_value: int):
    val = np.random.normal(mu, std)
    if val >= mu:
        step_value += 1
    else:
        step_value -= 1
    return step_value

def sample_from_distribution(name, a, b, prob_left, step_value, mu, std):
    if name == "normal".lower():
        return normal_decision(mu, std, step_value)
    elif name == "binomial".lower():
        return binomial_decision(prob_left, step_value)
    elif name == "uniform".lower():
        return uniform_decision(a, b, prob_left, step_value)

class CreatePheromoneScores:
    
    def _compute_direction_score(self, matrix: np.array, num_step: int, food_matrix: GetFoodMatrix):
        right_slice = food_matrix.slice_matrix(matrix, num_step, "right")
        left_slice = food_matrix.slice_matrix(matrix, num_step, "left")
        right_slice_sum = np.sum(right_slice)
        left_slice_sum = np.sum(left_slice)
        return right_slice_sum, left_slice_sum
    
    def _binomial_distribution(self):
        from_binomial = np.random.binomial(1, self.p_left)
        if from_binomial == 1:
            return 1
        else:
            return -1
    
    def compute_probabilities(self, matrix: np.array, num_step: int, food_matrix: GetFoodMatrix) -> Tuple[float, float]:
        pheromone_sum_right, pheromone_sum_left = self._compute_direction_score(matrix, num_step, food_matrix)
        self.p_right = (
            5 + np.square(pheromone_sum_left)
            / (
                (5 + np.square(pheromone_sum_left))
                + (5 + np.square(pheromone_sum_right))
            )
        )
        self.p_right = self.p_right/10
        self.p_left = 1 - self.p_right
        print(
            f"self.p_left: {self.p_left} \t self.p_right: {self.p_right}"
        )
        return self._binomial_distribution()
    
        
    
#num_ants, prob_left, NUM_STEPS, x, step_value, x_axis = 100, .2, 1000, 0, 0, np.linspace(1, NUM_STEPS, NUM_STEPS)
#a, b = 0, 1 # uniform
#mu, sigma = 0, .5 # Normal
#all_ants, lst_values = [], []
#plt.figure(figsize=(10,12))
#DISTRIBUTION = "uniform"
#x_axis = np.linspace(1, NUM_STEPS, NUM_STEPS)

#for ant in range(num_ants):
#    counter = 0
#    for step in range(NUM_STEPS):
#        step_value = sample_from_distribution(DISTRIBUTION, a, b, prob_left, step_value, mu, sigma)
#        if counter == 0:
#            counter += 1
#            step_value = (NUM_STEPS) / 2
#        lst_values.append(step_value)
#    all_ants.append(lst_values)
#    lst_values = []

#for i in range(len(all_ants)):
#    plt.scatter(all_ants[i], x_axis, s=3)