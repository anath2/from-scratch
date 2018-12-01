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


# Neural network

def make_network(layers_dims, activation_funcs):
    pass


# Network layer:

def make_layer(layer_vect: int, weight_matrix: np.array, activation_func: Callable):
    '''
    Create a layer of the neural network
    ARGS:
        layer_x: The values for nodes in the layer
        weights: A matrix of incoming weights to the network
        activation_func: Neural network activation function
    '''
    return (layer_vect, weight_matrix, activation_func)


def get_layer_value(layer: Tuple):
    '''
    Get number of nodes in the layer
    '''
    layer_vect = layer[0]
    return layer_vect


def get_layer_weights_matrix(layer: Tuple):
    '''
    Get weights going into the layer of the neural
    network
    '''
    weight_matrix = layer[1]
    return weight_matrix


def get_layer_activation_func(layer: Tuple):
    '''
    Get layer activation function
    '''
    activation_func = layer[2]
    return activation_func


# Train Network

def train_network(input_X: np.array, output_y: np.array, network: object, epoch_count: int):
    for epoch in epoch_count:
        output_layer = forward_prop(input_X, network)
        network = backward_prop(output_y, network)

    return network






def init_network(layers_dims: List, activation_funcs: List) -> Callable:
    '''
    Initialize a neural network setting random values for weights
    ARGS:
        layer_dims: Layers of neural network
        activation_funcs: List of activation function for each layer
    '''

    def _inner(input_vect):
        if len(activation_funcs) != len(layers_dims):
            raise NNException('Mismatch dimensions activation_funcs and layer dimensions')

        curr_layer = input_vect
        for layer_size, activation_func in zip(layers_dims, activation_funcs):
            weights_mat = np.random.randn(layer_size, len(input_vect))
            bias_vect = np.random.randn(1, layer_size)
            next_layer = calculate_next_layer(weights_matrix, input_vect, bias_vect, activation_func)

        return next_layer # Returns final layer

    return _forward_propagation


def calculate_next_layer(
        weights_matrix: np.array,
        input_vector: np.array,
        bias_vector: np.array,
        activation_func: Callable
) -> np.array:
    '''
    Calculates the next layer of a neural network
    '''
    weighted_sum = weights_matrix.dot(input_vector)
    weighted_sum = weighted_sum + bias_vector
    activation_func_vectorized = np.vectorize(activation_func)
    next_layer = activation_func_vectorized(weighted_sum)
    return next_layer


def make_node(weight_count: int, activation_func: Callable) -> Tuple[np.array, Callable]:
    '''Creates a node of the neural network'''
    weights = np.random.randn(weight_count)
    return (weights, activation_func)
