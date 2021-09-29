import numpy as np
from numpy.linalg import eigh, norm
from scipy.constants import hbar, h, mu_0
from matplotlib import pyplot as plt
from numpy.random import default_rng
from itertools import product

from quantum import Op

class MonteCarloAverage:
    def __init__(self):
        return