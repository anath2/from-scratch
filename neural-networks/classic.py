'''
Back propagation neural network written from scratch
Feed forward:

sigma(W2 * sigma (W1 * x + b1) + b2)

Back propagation:
Adjust values for weights with gradient descent

'''

from typing import Callable, List, Tuple
import numpy as np


class NNException(Exception):
    '''General exception raised for neural networks'''


def train(train_X, train_y):
    pass


def test(test_X, test_y):
    pass


def forward_propagation(training_X, network: Tuple):
   pass


def backward_propagation(training_y: np.array, network: Tuple):
    pass


def make_neural_network(input_data: np.array, layer_dims: List, activation_funcs: List) -> Tuple[np.array, List]:
    '''
    Create a neural network with given input, ouput and hidden layer
    ARGS:
        input_data: Refers to the input layer of the neural network
        layers_dims: A list representing the number of nodes in each layer of nn
        activation_funcs: A list of activation functions
    '''
    if len(activation_funcs) != len(layer_dims):
        raise NNException('Length mismatch: length of activation not same as network size')
    if len(input_data) != layer_dims[0]:
        raise NNException('Dimensions of the input data should match first elem of layers_dims')
    
    layer_dims = len(input_data) + layer_dims # Add input dimension to layers
    layers = []
    for ind, node_count in enumerate(layer_dims):
        if ind == 0:
            continue
        weight_count = layer_dims[ind - 1]
        activation_func = activation_funcs[ind]
        layer = make_layer(node_count, weight_count, activation_func)
        layers.append(layer)

    return (input_data, layers)


def make_layer(node_count, weight_count, activation_func: Callable) -> List:
    '''
    Create a layer of the neural network
    ARGS:
        node_count: Number of nodes in the layer
        weight_count: Number of incoming values
        activation_func: Neuron activation function
    ''' 
    layer_nodes = [make_node(weight_count, activation_func) for n in node_count]
    return layer_nodes


def make_node(weight_count: int, activation_func: Callable) -> Tuple[np.array, Callable]:
    '''Creates a node of the neural network'''
    weights = np.random.randn(weight_count)
    return (weights, activation_func)
