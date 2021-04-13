import numpy as np
import pandas as pd

def min_max_scale(array):
    return (array - array.min())/(array.max() - array.min())

def get_input(filename, attributes):
    df = pd.read_csv(filename)
    y = np.array(df['SalePrice'])
    x = np.array(
        [min_max_scale(df[a]) for a in attributes]
    )
    return x,y