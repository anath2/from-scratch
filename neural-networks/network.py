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
    '''
    # Input nodes are the ones where the input symbol in
    # Not mentioned elsewhere in any of the outputs
    # Output nodes are the ones where the output is not
    # connected to any of the input nodes throughout the network
    # The rest of the nodes are hidden nodes
    pass


def feed_forward(network: List) -> np.array:
    '''
    Feeds forward the network and calculates the result
    '''
    pass
