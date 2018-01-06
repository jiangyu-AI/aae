from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from autoencoder_models.GumbelAutoencoder import GumbelAutoencoder as Autoencoder

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = mnist.train.images, mnist.test.images
#X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

anneal_rate = 0.0003
min_temperature = 0.5
tau = 5.0#, name="temperature")
#tau = K.Variable(5.0, name="temperature")

autoencoder = Autoencoder(
    n_input=784,
    n_hidden=200,
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

for epoch in range(training_epochs):
    tau = np.max([tau * np.exp(- anneal_rate * epoch), min_temperature])
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs, tau)
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%d,' % (epoch + 1),
              "Cost:", "{:.9f}".format(avg_cost))

print("Total cost: " + str(autoencoder.calc_total_cost(X_test, tau)))

import pathlib
pathlib.Path('model_dir').mkdir(parents=True, exist_ok=True) 
save_path = autoencoder.save('model_dir')
print("Model saved in file: %s" % save_path)
