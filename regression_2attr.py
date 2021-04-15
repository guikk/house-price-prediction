import numpy as np
import sys
from util import get_input

def main():
    try:
        input_file = sys.argv[1]
        n = int(sys.argv[2])
    except:
        raise Exception(f'Wrong input format: {sys.argv} (expected 3)')
    x,y = get_input(input_file,['GrLivArea','OverallQual'])
    # print(compute_cost(np.random.rand(3,1), x, y))
    theta, cost_graph, theta_progress = gradient_descent(x, y, learning_rate=1e-7, num_iterations=n)
    print(f"theta_0: {theta[0]}")
    print(f"theta_1: {theta[1]}")
    print(f"theta_2: {theta[2]}")
    print(f"Erro quadratico medio: {cost_graph[-1]}")
    
    return 0

def compute_cost(theta, x, y):

    h = theta[2]*x[1] + theta[1]*x[0] + theta[0]
    quad_error = np.power(h-y, 2)
    mean = np.sum(quad_error)/y.size

    return mean

def step_gradient(theta, x, y, alpha):
    theta_new = [0,0,0]

    error = (theta[2]*x[1] + theta[1]*x[0] + theta[0]) - y
    
    dt0 = (2/x.size) * np.sum(error)
    dt1 = (2/x.size) * np.sum(np.multiply(x[0], error))
    dt2 = (2/x.size) * np.sum(np.multiply(x[1], error))

    theta_new[0] = theta[0] - alpha * dt0
    theta_new[1] = theta[1] - alpha * dt1
    theta_new[2] = theta[2] - alpha * dt2

    return theta_new

def gradient_descent(x, y, starting_theta=[0,0,0], learning_rate=0.005, num_iterations=10):

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