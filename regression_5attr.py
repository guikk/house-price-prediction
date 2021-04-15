import numpy as np
import sys
from util import get_input

def main():
    try:
        input_file = sys.argv[1]
        n = int(sys.argv[2])
    except:
        raise Exception(f'Wrong input format: {sys.argv} (expected 3)')
    x,y = get_input(input_file,['GrLivArea', 'OverallQual', 'OverallCond', 'GarageArea', 'YearBuilt'])
    theta, cost_graph, theta_progress = gradient_descent(x, y, learning_rate=1e-7, num_iterations=n)
    print(f"theta_0: {theta[0]}")
    print(f"theta_1: {theta[1]}")
    print(f"theta_2: {theta[2]}")
    print(f"theta_3: {theta[3]}")
    print(f"theta_4: {theta[4]}")
    print(f"theta_5: {theta[5]}")
    print(f"Erro quadratico medio: {cost_graph[-1]}")
    return 0

def compute_cost(theta, x, y):

    estimate = x[4]*theta[5] + x[3]*theta[4] + x[2]*theta[3] + x[1]*theta[2] + x[0]*theta[1] + theta[0]
    quad_error = np.power(estimate-y, 2)
    mean = np.sum(quad_error)/x.size

    return mean

def step_gradient(theta, x, y, alpha):
    theta_new = [0,0,0,0,0,0]

    error = (x[4]*theta[5] + x[3]*theta[4] + x[2]*theta[3] + x[1]*theta[2] + x[0]*theta[1] + theta[0]) - y
    
    dt0 = (2/x.size) * np.sum(error)
    dt1 = (2/x.size) * np.sum(np.multiply(x[0], error))
    dt2 = (2/x.size) * np.sum(np.multiply(x[1], error))
    dt3 = (2/x.size) * np.sum(np.multiply(x[2], error))
    dt4 = (2/x.size) * np.sum(np.multiply(x[3], error))
    dt5 = (2/x.size) * np.sum(np.multiply(x[4], error))

    theta_new[0] = theta[0] - alpha * dt0
    theta_new[1] = theta[1] - alpha * dt1
    theta_new[2] = theta[2] - alpha * dt2
    theta_new[3] = theta[3] - alpha * dt3
    theta_new[4] = theta[4] - alpha * dt4
    theta_new[5] = theta[5] - alpha * dt5

    return theta_new

def gradient_descent(x, y, starting_theta=[0,0,0,0,0,0], learning_rate=0.005, num_iterations=10):

    theta = starting_theta
    
    cost_graph = [compute_cost(theta, x, y)]
    
    theta_progress = []
    
    for _ in range(num_iterations):
        cost_graph.append(compute_cost(theta, x, y))
        theta = step_gradient(theta, x, y, alpha=learning_rate)
        theta_progress.append(theta)
        
    return theta, cost_graph, theta_progress

if __name__ == "__main__":
    main()