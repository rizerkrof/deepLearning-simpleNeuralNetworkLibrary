#!/usr/bin/env python3
import sys
sys.path.insert(0, sys.path[0]+'/../')

import numpy as np
from src.library import DeepNeuralNetwork

def setupTestingDNNAndSample():
    deepNeuralNetwork = DeepNeuralNetwork(n_neurons=[10, 5, 4, 3, 2])
    X_train = np.array([
        np.array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8]),
        np.array([0, 1, 3, 3, 5, 5, 7, 7, 9, 9]),
    ])
    y_train = np.array([
        np.array([0, 1]),
        np.array([1, 0]),
    ])
    return (deepNeuralNetwork, X_train, y_train)

def tests_ForwardPassShape():
    deepNeuralNetwork, X_train, y_train = setupTestingDNNAndSample()
    output = deepNeuralNetwork.forward_pass(X_train[1])
    assert output.shape == y_train[1].shape

def tests_BackwardPassShape():
    deepNeuralNetwork, X_train, y_train = setupTestingDNNAndSample()
    output = deepNeuralNetwork.forward_pass(X_train[1])
    updates = deepNeuralNetwork.backward_pass(y_train[1], output)
    for key, value in updates.items():
        if not np.isscalar(value):
            assert value.shape == deepNeuralNetwork.params[key].shape, f"param {key} shape: {deepNeralNetwork.params[key].shape} != Update shape: {value.shape}"
        else:
            assert np.isscalar(deepNeuralNetwork.params[key]), f"param {key} has shape {deepNeuralNetwork.params[key].shape} but update is scalar={value}"
