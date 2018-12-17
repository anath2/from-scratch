'''
Back propagation neural network written from scratch
Feed forward:

sigma(W2 * sigma (W1 * x + b1) + b2)

Back propagation:
Adjust values for weights with gradient descent

'''

from collections import namedtuple
from typing import Callable, List, Tuple
import numpy as np


class NNException(Exception):
    '''General exception raised for neural networks'''


# Neural network

def make_network(layer_dims: List, activation_funcs: Callable) -> List:
    '''
    Neural network constructor. A neural network is represented by
    pairs of layer dimensions and their corresponding activation
    functions
    ARGS:
        layer_dims: Dimensions of each layer in the neural network
        activation_funcs: Activations associated with each of the layers
    RETURNS:
        network: Neural network representation
    '''
    validate_network_params(layers_dims, activation_funcs)
    network = list(zip(layers_dims, activation_funcs))
    return network


def validate_network_params(layer_dims: List, activation_funcs: List[Callable]):
    '''
    Validate params to neural network parameters
    ARGS:
        layer_dims: Dimensions of layers in network
        activation_funcs: Activation function for each layer
    '''
    try:
        assert len(layers_dims) > 1
        assert len(layers_dims) == len(activation_funcs)
        assert all(isinstance(dim, int) for dim in layer_dims)
    except AssertionError as err:
        raise NNException('Network params validation error {}'.format(err))

# Network Layer

def make_layer(input_vect: object, layer_node_count: int, activation_func: Callable) -> object:
    '''
    Create a layer of the neural network
    ARGS:
        input_vect: Input vector to the layer
        layer_node_count: Number of neurons in the layer
        activation_func: Function to calculate activation of neurons in the layer
    '''
    input_dims = len(input_vect)
    weight_matrix = np.random.randn(layer_size, input_dims)
    bias_vector = np.random.randn(layer_size)
    layer_vect = calculate_next_layer(weight_matrix, previous_layer, bias_vector, activation_func)
    NetworkLayer = namedtuple('NetworkLayer', 'layer_vect, weight_matrix, activation_func')
    return NetworkLayer(layer_vect=layer_vect, weight_matrix=weight_matrix, activation_func=activation_func)


def calculate_next_layer(
        weights_matrix: np.array,
        input_vector: np.array,
        bias_vector: np.array,
        activation_func: Callable
) -> np.array:
    '''
    Calculates the next layer of a neural network
    ARGS:
        weights_matrix: Matrix with rows as input weights to each neuron
        input_vector: Input vector to the layer
        bias_vector: Biases to each neuron of the network
        activation_func: Activation function for each neuron in the layer
    '''
    weighted_sum = weights_matrix.dot(input_vector)
    weighted_sum = weighted_sum + bias_vector
    activation_func_vectorized = np.vectorize(activation_func)
    next_layer = activation_func_vectorized(weighted_sum)
    return next_layer


# Train Network

def train_network(input_X: np.array, output_y: np.array, network: object, epoch_count: int):
    for epoch in epoch_count:
        output_layer = forward_prop(input_X, network)
        network = backward_prop(output_y, network)

    return network
