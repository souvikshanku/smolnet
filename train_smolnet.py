"""Train and save smolnet in a pickle file.
"""
import gzip
import pickle

import numpy as np

from smolnet import Network


with gzip.open('data/mnist.pkl.gz', "rb") as f:
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')

def vectorize(j):
    """Return vectorized binary array with 1 at j-th position.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

train_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
train_labels = [vectorize(y) for y in training_data[1]]

training_data = list(zip(train_inputs,  train_labels))

test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
test_labels = [vectorize(y) for y in test_data[1]]

test_data = list(zip(test_inputs,  test_labels))



smolnet = Network((784, 64, 32, 10))
smolnet.learning_rate = .8

smolnet.train(
    training_data=training_data,
    epochs=20,
    batch_size=32,
    test_data=test_data
)

with open('my_smolnet.pkl', 'wb') as  f:
    pickle.dump(smolnet, f)
