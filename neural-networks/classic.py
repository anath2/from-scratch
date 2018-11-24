'''
Back propagation neural network written from scratch
Feed forward:

sigma(W2 * sigma (W1 * x + b1) + b2)

Back propagation:
Adjust values for weights with gradient descent

'''

from typing import Callable
import numpy as np


def make_neural_network(input_array: np.array):

    '''
    Create a neural network with given input, ouput and hidden layer
    '''
    pass


def make_layer(node_count, activation_func: Callable):
    '''
    Create a layer of the neural network
    '''
    pass


def make_node(in_weights: np.array, activation_func: Callable):
    '''Creates a node of the neural network'''
    pass
