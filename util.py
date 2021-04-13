import numpy as np

def min_max_scaling(array: np.ndarray) -> np.ndarray:
    return (array - array.min())/(array.max() - array.min())

def compute_cost_linear(w, x, y):

    estimate = x*w[1] + w[0]
    quad_error = np.power(estimate-y, 2)
    mean = np.sum(quad_error)/x.size

    return mean