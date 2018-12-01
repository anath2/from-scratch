'''
Neuron activation functions:
    RELU
    softmax
    sigmoid
'''

import math
import numpy as np


def relu(x):
    '''f(x) -> x'''
    return max(0, x)


def softmax(x_arr: np.array) -> np.array:
    '''
    Generally represents a catagorical distribution
    '''
    exp_z = np.array([math.exp(elem) for elem in x_arr])
    result = exp_z / exp_z.sum()
    return result


def tanh(x):
    '''
    Hyperbolic tangent functions
    '''
    num = math.exp(2 * x) - 1
    denom = math.exp(2 * x) + 1
    return num / denom


def logistic(x):
    '''
    Logistic function.
    '''
    return 1.0 / (1 + math.exp(-1 * x))
