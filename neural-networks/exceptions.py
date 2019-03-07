'''
Exception classes for neural network
'''

class NeuralNetworkError(Exception):
    ''''
    Base exception class for all neural network
    related exceptions
    '''


class NeuronError(NeuralNetworkError):
    '''
    Exception occurs when neuron tries to process input
    '''
    pass
