import numpy as np
import os

def write_bin(mat, path):
    mat.astype(np.float64).tofile(path)