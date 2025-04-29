from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

class GetFoodMatrix:
    
    def _get_left_ahead(self, matrix: np.array, num_step: int) -> np.array:
        """
        Creates submatrix from original matrix to get slice the fields that ants can eat to the left
        Parameters:
        -----------
        
        matrix: np.array - matrix to be sliced
        num_step: int - which step is it (slicing condition)
        
        Returns:
        --------
        matrix: np.array - returns sliced matrix
        """
        index_one_step_ahead = num_step + 1
        return matrix[
            0:index_one_step_ahead, 0:index_one_step_ahead
        ]

    def _get_right_ahead(self, matrix: np.array, num_step: int) -> np.array:
        """
        Creates submatrix from original matrix to get slice the fields that ants can eat to the right
        Parameters:
        -----------
        
        matrix: np.array - matrix to be sliced
        num_step: int - which step is it (slicing condition)
        
        Returns:
        --------
        matrix: np.array - returns sliced matrix
        """
        index_one_step_ahead = num_step + 1
        return matrix[
            0:index_one_step_ahead, -index_one_step_ahead:
        ]
    
    def slice_matrix(self, matrix: np.array, num_step: int, slice_type: str) -> np.array:
        """
        Creates submatrix from original matrix by specifying slice type
        Parameters:
        -----------
        
        matrix: np.array - matrix to be sliced
        num_step: int - which step is it (slicing condition)
        slice_type - which slice should be made
        
        Returns:
        --------
        matrix: np.array - returns sliced matrix
        """
        if slice_type == "right".lower():
            return self._get_left_ahead(matrix, num_step)
        else:
            return self._get_right_ahead(matrix, num_step)

def get_direction(matrix: np.array, num_step: int, food_matrix: GetFoodMatrix):
    right_slice = food_matrix.slice_matrix(matrix, num_step, "right")
    left_slice = food_matrix.slice_matrix(matrix, num_step, "left")
    right_slice = np.sum(right_slice)
    left_slice = np.sum(left_slice)
    print(f"left_slice: {left_slice}, right_slice: {right_slice}")
    if right_slice > left_slice:
        return 1
    else:
        return -1
    
class CreatePheromoneScores:
    
    def _compute_direction_score(self, matrix: np.array, num_step: int, food_matrix: GetFoodMatrix):
        right_slice = food_matrix.slice_matrix(matrix, num_step, "right")
        left_slice = food_matrix.slice_matrix(matrix, num_step, "left")
        right_slice_sum = np.sum(right_slice)
        left_slice_sum = np.sum(left_slice)
        return right_slice_sum, left_slice_sum
    
    def _binomial_distribution(self, prob: float):
        from_binomial = np.random.binomial(1, prob)
        if from_binomial == 1:
            return 1
        else:
            return -1
    
    def compute_probabilities(self, matrix: np.array, num_step: int, food_matrix: GetFoodMatrix) -> Tuple[float, float]:
        pheromone_sum_right, pheromone_sum_left = self._compute_direction_score(matrix, num_step, food_matrix)
        p_l = 100 # konstanta
        self.p_right = (
            5 + np.square(pheromone_sum_left)
            / (
                (5 + np.square(pheromone_sum_left))
                + (5 + np.square(pheromone_sum_right))
            )
        )
        move_per_step_prob = .5 + .5 * np.tanh(
                (pheromone_sum_right + pheromone_sum_left) / p_l - 1
            )
        move_per_step_prob = move_per_step_prob/10    
        self.p_right = self.p_right/10
        self.p_left = 1 - self.p_right
        print(
            f"move_to_step_from_binomial: {self._binomial_distribution(move_per_step_prob)} \t move_per_step_prob: {move_per_step_prob}"
        )
        return self._binomial_distribution(self.p_left), self._binomial_distribution(move_per_step_prob)
    
    


'''
NUM_STEPS = 100
num_ants, prob_left, x, step_value, x_axis = 1000, .2, 0, 0, np.linspace(1, NUM_STEPS, NUM_STEPS)
a, b = 0, 1 # uniform
mu, sigma = 0, .5 # Normal
all_ants, lst_values = [], []
plt.figure(figsize=(10,12))
DISTRIBUTION = "uniform"
food_matrix_slicer = GetFoodMatrix()
x_axis = np.linspace(1, NUM_STEPS, NUM_STEPS)

for ant in range(num_ants):
    matrix = np.random.rand(NUM_STEPS, NUM_STEPS)
    counter = 0
    for step in range(NUM_STEPS):
        step_value += get_direction(matrix, step, food_matrix_slicer)
        if counter == 0:
            counter += 1
            step_value = (NUM_STEPS) / 2
        lst_values.append(step_value)
    all_ants.append(lst_values)
    lst_values = []

for i in range(len(all_ants)):
    plt.scatter(all_ants[i], x_axis, s=3)  
'''