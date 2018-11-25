'''
Neuron activation functions:
    RELU
    softmax
    sigmoid
'''
import math
import numpy as np


def relu(x):
    pass


def softmax(x_arr: np.array) -> np.array:
    '''
    Generally represents a catagorical distribution
    '''
    exp_z = np.array([math.exp(elem) for elem in x_arr])
    result = exp_z / exp_z.sum()
    return result


def sigmoid(x):
    pass

