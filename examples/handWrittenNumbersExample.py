#!/usr/bin/env python3
import sys
sys.path.insert(0, sys.path[0]+'/../')

from sklearn.datasets import fetch_openml
from src.library.deepNeuralNetwork import DeepNeuralNetwork
from src.tools.plotLearningCurves import plotLearningCurves

from sklearn.model_selection import train_test_split
import numpy as np

print('fetching data...')
data_bunch = fetch_openml('mnist_784', as_frame=False)

print('normalizing data...')
X = (data_bunch.data/255).astype('float32')
y = data_bunch.target.astype('uint32')

print('splinting testing/validation data...')
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print('one hot encoding data...')
def one_hot(a):
    num_classes = np.max(a) + 1
    return np.eye(num_classes, dtype="uint32")[a.reshape(-1)]

Y_train = one_hot(y_train)
Y_val = one_hot(y_val)

np.random.seed(42)
dnn = DeepNeuralNetwork(n_neurons=[784, 128, 64, 10])

print('fiting network...')
history = dnn.fit(X_train[:10000, :], Y_train[:10000], X_val, Y_val, epochs=5, batch_size=16, l_rate=1e-2)
