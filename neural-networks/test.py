''' Modul efor running tests on neural networks'''

import pandas as pd
from . import classic

def mnist_test():
    ''' Classify mnist hand written digits'''
    training_data = pd.read_csv('./test-data/mnist_X.csv')
    trainig_labels = pd.read_csv('./test-data/mnist_y.csv')
    
    test_data = pd.read_csv('./test_data_X.csv')
    test_labels = pd.read_csv('./test_data_y.csv')

    network = classic.make_neural_network(
        training_data, 
        [64, 64, 10], 
        [activation.relu, activation.sigmoid, activation.softmax]
    )
   
    network = classic.train(network)
    accuracy = classic.test(test_data, test_labels, network)
    return accuracy
