import numpy as np
from util import get_input

def main():
    x,y = get_input(['GrLivArea'])
    theta, cost_graph, theta_progress = gradient_descent(x, y, learning_rate=1e-7, num_iterations=10)
    print(f"theta_0: {theta[0]}")
    print(f"theta_1: {theta[1]}")
    print(f"Erro quadratico medio: {cost_graph[-1]}")
    return 0

def compute_cost(theta, x, y):

    estimate = x*theta[1] + theta[0]
    quad_error = np.power(estimate-y, 2)
    mean = np.sum(quad_error)/x.size

    return mean

def step_gradient(theta, x, y, alpha):
    theta_new = [0,0]

    error = (x * theta[1] + theta[0]) - y
    
    dt0 = (2/x.size) * np.sum(error)
    dt1 = (2/x.size) * np.sum(np.multiply(x, error))

    theta_new[0] = theta[0] - alpha * dt0
    theta_new[1] = theta[1] - alpha * dt1

    return theta_new

def gradient_descent(x, y, starting_theta=[0,0], learning_rate=0.005, num_iterations=10):

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