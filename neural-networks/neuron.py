'''
Neuron and operations on it
'''

from typing import Callable
from collections import namedtuple

import numpy as np
from . import exceptions


Neuron = namedtuple('Neuron', 'weights, bias')


def feed_forward(neuron: Neuron, input_arr: np.array, activation_func: Callable):
    '''
    ARGS:
       neuron: The neuron to which the data is fed
       inputs: Inputs to the neuron

    Creates a feed forward network
    '''
    if len(input) ! = len(neuron.weights):
        message = 'Unequal lengths - {weights} and input - {input_arr}'
        raise exceptions.NeuronError(message.format(weights=neuron.weights, input_arr=input_arr))

    return activation_func(np.dot(inputs, neuron.weights) + neuron.bias)
