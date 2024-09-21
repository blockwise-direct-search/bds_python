import numpy as np

def is_scalar(value):
    r"""
    Verify if the input variable is a real scalar 
    (integer, float or NumPy scalar)
    """
    return isinstance(value, (int, float, np.number))