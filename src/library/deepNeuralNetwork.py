#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import time
from collections import defaultdict
from numpy.typing import ArrayLike
from typing import List, Dict, Tuple

from tqdm.notebook import tqdm

from .activationFunctions import sigmoid
from .activationFunctions import relu
from .activationFunctions import softmax

class DeepNeuralNetwork():
    def __init__(self, n_neurons:list):
        """AI is creating summary for __init__
            The class constructor has only one argument, that determines the
            size of each layer.
            It initializes network parameters according to these sizes
        Args:
            n_neurons (list): a list of integers containing the quantity of
            neurons in each layer. The lenght of the list corresponds to the number of layers (including the input layer).
        """
        self.n_neurons = n_neurons
        self.n_layers = len(n_neurons)
        # we save all parameters of the neural network in this dictionary
        self.params = self._initialization()

    def _initialization(self) -> Dict:
        """Initializes parameters for the network and gathers them
        in a dictionnary. This method is called in the class constructor.

        Weights for layer i are named Wi
        biasfor layer i are named bi

        Returns:
            dict: dictionnary of parameters in the network. Weights for layer i
            are named Wi (W1, W2, etc). Bias for layer i is named bi (b1, b2,
            etc)
        """
        params = {}
        for i in range(self.n_layers-1):
            wShape = (self.n_neurons[i+1], self.n_neurons[i])
            params['W'+str(i+1)] = np.random.randn(*wShape) * np.sqrt(1. / wShape[0])
            params['B'+str(i+1)] = np.zeros((self.n_neurons[i+1], ), dtype=int)
        return params

    def forward_pass(self, x) -> ArrayLike:
        """
        This functions computes the forward pass for a single sample x
        """
        params = self.params
        # input layer activations becomes sample
        params['Z0'] = x

        for i in range(1, self.n_layers):
          params['A'+str(i)] = params['W'+str(i)] @ params['Z'+str(i-1)] + params['B'+str(i)]
          params['Z'+str(i)] = relu(params['A'+str(i)]) if i!= self.n_layers-1 else softmax(params['A'+str(i)])

        output = params['Z'+str(self.n_layers-1)]
        return output

    def backward_pass(self, y, output) -> Dict:
        '''
        This is the backpropagation algorithm, for calculating the updates
        of the neural network's parameters.

        Note: There is a stability issue that causes warnings. This is
                caused  by the dot and multiply operations on the huge arrays.

                RuntimeWarning: invalid value encountered in true_divide
                RuntimeWarning: overflow encountered in exp
                RuntimeWarning: overflow encountered in square
        '''
        params = self.params
        change_params = {}

        grad = np.multiply(output-y, softmax(params['A'+str(self.n_layers-1)], derivative=True))
        change_params['W'+str(self.n_layers-1)] = np.outer(grad, params['Z'+str(self.n_layers-2)])
        change_params['B'+str(self.n_layers-1)] = grad
        for i in reversed(range(1, self.n_layers-1)):
            grad = np.multiply( np.transpose(params['W'+str(i+1)])@grad, relu(params['A'+str(i)], derivative=True))
            change_params['W'+str(i)] = np.outer(grad, params['Z'+str(i-1)])
            change_params['B'+str(i)] = grad

        return change_params

    def update_network_parameters(self, changes_params, l_rate):
        '''
        Update network parameters according to update rule from
        Stochastic Gradient Descent.

        θ = θ - η * ∇J(x, y),
            theta θ:            a network parameter (e.g. a weight w)
            eta η:              the learning rate
            gradient ∇J(x, y):  the gradient of the cost function,
                                i.e. the change for a specific theta θ
        '''
        for key, value in changes_params.items():
            self.params[key] = self.params[key] - l_rate * value

    def predict_proba(self, X):
        """
        This function computes predictions for a maxtrix of samples X
        by calling the forward_pass on each row.
        Returns a class distribution vector per sample
        """
        return np.array([self.forward_pass(x_i) for x_i in X])

    def predict(self, X):
        """
        This function computes predictions for a maxtrix of samples X
        by calling the forward_pass on each row
        Returns a class id per sample
        """
        return np.array([np.argmax(self.forward_pass(x_i)) for x_i in X], dtype='uint32')

    def score(self, X, y):
        '''
        This function computes predictions for a maxtrix of samples X,
        then checks if the index of the maximum value
        in the output equals the index in the label y.
        This indicator vector of correct/incorrect predictions is
        averaged to obtain the accuracy

        y may be a 1d array of labels or a one-hot encoded matrix of label indicators
        '''
        if y.ndim == 1:
            y_true = y
        elif y.ndim == 2:
            y_true = np.argmax(y, axis=1)

        return np.mean(y_true == self.predict(X))

    def fit(self, X_train:ArrayLike, Y_train:ArrayLike,
            X_val:ArrayLike, Y_val:ArrayLike,
            epochs=10, batch_size=16, l_rate=0.001) -> Dict:
        """This function fits the model parameters to the training data,
        while monitoring the accuracy score on the validation data.

        Y arrays should be one-hot encoded

        Args:
            X_train (ArrayLike): Array with training input samples
            Y_train (ArrayLike): One-hot encoded array of training set targets
            X_val (ArrayLike): Array with validation samples
            Y_val (ArrayLike): One-hot encoded array of validation set targets
            epochs (int, optional): number of epochs for SGD. Defaults to 10.
            l_rate (float, optional): learning rate for SGD. Defaults to 0.001.

        Returns:
            Dict: dict containing the history of accuracy and loss metrics
            across training epochs
        """
        history = defaultdict(list)
        for iteration in range(epochs):
            start_time = time.time()
            # split sample indexes in batches
            batch_indexes = np.array_split(np.arange(X_train.shape[0]), batch_size)
            # Loop over batches of trainng samples
            # tqdm allows us to show a progress bar
            for batch_i, idx in tqdm(enumerate(batch_indexes),
                                    desc=f"epoch {iteration+1} progress",
                                    total=len(batch_indexes)):
                # this dictionnary will store the updates computed
                # over all samples in the batch
                # values get updated after each sample is processed
                batch_changes = defaultdict(int)
                # loop over batch samples
                for x, y in zip(X_train[idx], Y_train[idx]):
                    # compute forward pass, backward pass,
                    output = self.forward_pass(x)
                    changes_to_w = self.backward_pass(y, output)
                    # sum updates in batch_changes dictionary
                    for key, value in changes_to_w.items():
                         batch_changes[key] += value/batch_size

                # apply batched GD updates
                self.update_network_parameters(batch_changes, l_rate)

            # store performances on train and validation
            # for learning curves later
            for split, X, Y in zip(['train', 'val'],
                                [X_train, X_val],
                                [Y_train, Y_val]):
                y_pred_proba = self.predict_proba(X)
                y_pred = np.argmax(y_pred_proba, axis=1)
                loss = log_loss(Y, y_pred_proba)
                y_true = np.argmax(Y, axis=1)
                accuracy = accuracy_score(y_true, y_pred)

                history[split+'_loss'].append(loss)
                history[split+'_accuracy'].append(accuracy)
            # Print stats about this epoch
            acc = history['val_accuracy'][-1]
            print(
                f'Epoch: {iteration + 1}, Time Spent: {time.time() - start_time:.2f}s, Val Accuracy: { acc * 100:.2f}%')

        return history
