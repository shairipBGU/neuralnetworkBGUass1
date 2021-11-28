import numpy as np
import math

from numpy.random import randn


def f(teta, x, sigmoid):
    bias = teta[:, -1]
    weights = teta[:, 0:-1]
    teta_x = weights @ x
    x_plus_bias = teta_x + bias[:, np.newaxis]
    return sigmoid(x_plus_bias) # todo do a remap of the bias


def main():
    # define hyper parameters (number of layers, size of each layer)
    input_features = 2 # depends on the data set we are running on
    layer_widths = [input_features, 5, 5, 5] # todo figure out how to connect to the end layer - the softmax

    tetas = []
    # weights initialization
    for l in range(len(layer_widths) - 1):
        n1 = layer_widths[l]
        n2 = layer_widths[l + 1]
        teta_l = randn(n2, n1) / math.sqrt(n1 * n2) # initialize n1Xn2 weights matrix between layer l and layer l+1 with norm 1
        biases = np.zeros(n2).reshape(n2, 1) # initialize the biases to zero
        teta_l = np.hstack((teta_l, biases))
        # print(teta_l)
        np_matrix = np.asarray(teta_l)
        tetas.append(teta_l)
        # print(np_matrix.size)
    # print(len(tetas))

    # forward pass
    # assume x_matrix as input
    x_matrix = np.asarray([[1, 2, 3],
                          [2, 3, 4]])

    x_save_for_later = []
    x_i = x_matrix
    for teta in tetas:
        x_i = f(teta, x_i, np.tanh)

        x_save_for_later.append(x_i.copy())

if __name__ == '__main__':
    main()