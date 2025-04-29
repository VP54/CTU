from typing import List, Tuple, Generator
from itertools import compress
import numpy as np

def restrict_edge_values(NUM_PARTICLES: int, *coords: Tuple[int, int]) -> bool:
    """
    Calculates probability that particle given coordinates will spin
    Paramters: 
    ----------
    NUM_PARTICLES: int - maximum grid limit
    coords: Tuple[int, int] - [x, y] coordinates of particle
    """
    #NUM_PARTICLES = 10
    for coord in coords:
        if (
            (coord[0] in [-1, NUM_PARTICLES]) 
            or (coord[1] in [-1, NUM_PARTICLES])
        ):
            yield False
        else:
            yield True

def get_spin_value(spin_values_matrix: np.matrix, coords: List[Tuple[int, int]]) -> Generator[int, int, int]:
    """
    Gets spin value from coordinates and matrix with spin values
    Paramters: 
    ----------
    spin_values_matrix: np.matrix - matrix of spin_values for each particle (node)
    coords: Tuple[int, int] - [x, y] coordinates of particle
    """
    for coord in coords:
        x, y = coord[0], coord[1]
        print(x, y)
        yield spin_values_matrix[int(x), int(y)]

def sum_neighbors(x: int, y: int, spin_values_matrix: np.matrix, NUM_PARTICLES: int) -> int:
    """
    Sums spin values in neighbouroughood of particle located by x, y
    Paramters: 
    ----------
    x: int - x_coordinate of particle
    y: int - y_coordinate of particle
    spin_field: np.matrix - matrix of spin_values for each particle (node)
    NUM_PARTICLES: int - maximum grid limit
    """
    node_up, node_to_left, node_to_right, node_down = (x, y+1), (x-1, y), (x+1, y), (x, y-1)
    lst_nodes = [node_up, node_to_left, node_to_right, node_down]
    is_summable = restrict_edge_values(NUM_PARTICLES, *lst_nodes)
    nodes_to_be_summed = list(compress(lst_nodes, is_summable))
    spin_value_of_nodes = get_spin_value(spin_values_matrix, nodes_to_be_summed)
    return - np.sum(list(spin_value_of_nodes))

def probability_spin(BETA: float, sum_neighbors: int) -> float:
    """
    Calculates probability that particle given coordinates will spin
    Paramters: 
    ----------
    BETA: float - coefficient describing influence of energy function
    """
    V = 4 # number of vertexes |V|
    return (
        V**-1 * np.exp(BETA * sum_neighbors) 
        / ((np.exp(BETA * sum_neighbors) + np.exp(- BETA * sum_neighbors)))
    )
    
def should_turn(delta_E: float) -> bool:
    """
    Determines if particle should spin
    Paramters:
    ----------
    deltaE: float - 
    """
    if delta_E == 0:
        return np.random.choice([True, False]) 
        # if the Energy difference is equal to zero, there is 0.5 probability of spin, so randomly uniformly assign True/False
    elif delta_E < 0:
        return True

BETA = 0.5
NUM_SIMULATIONS = 100
NUM_PARTICLES = 10

x_coords = np.linspace(0, NUM_PARTICLES-1, NUM_PARTICLES, int)
y_coords = np.linspace(0, NUM_PARTICLES-1, NUM_PARTICLES, int)
matrix = np.ones([10, 10])

print(matrix)

for xx in range(matrix.shape[0]):
    for yy in range(matrix.shape[1]):
        matrix[xx, yy] = np.random.choice([-1, 1])

for simulation in range(NUM_SIMULATIONS):
    x = np.random.choice(x_coords).astype(int)
    y = np.random.choice(y_coords).astype(int)
    sum_neighbors_value = sum_neighbors(x, y, matrix, NUM_PARTICLES)
    prob_spin = probability_spin(BETA, sum_neighbors_value)
    print(f"x: {x} \t y: {y} \t prob_spin: {prob_spin}")
    if should_turn(prob_spin):
        matrix[x, y] = - matrix[x, y]
        
