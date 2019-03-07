'''
Neural Network
'''

from typing import Dict
from collections import OrderedDict

import numpy as np

from . import neuron


def make_network(json: OrderedDict) -> List:
    '''
    Creates a network of neurons using a
    Example Network JSON:
    {
      'layer0' : {
        'neuron00': {
          weights: [0, 1],
          bias: 0,
          activation_func: 'relu'
        ...
      },
      {
       'layer1' : {
        'neuron10': {
          weights: [0, 1],
          bias: 0,
          activation_func: 'relu'
        ...
      }
    }
    '''
    pass


def feed_forward(network: List) -> np.array:
    '''
    Feeds forward the network and calculates the result
    '''
    pass
