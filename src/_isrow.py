import numpy as np
import pdb


def check_row_vector(x):
    x_is_row = False
    #pdb.set_trace()
    if isinstance(x, np.ndarray) and x.ndim == 1:
        x_is_row = True
    return x_is_row