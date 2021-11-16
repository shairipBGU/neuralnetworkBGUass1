import math
import numpy as np


# receives w - all weights (in matrix form?), j - the label we want to check zero based, x - the sample,
# returns a probability of how much the classifier is sure the X sample label(y) is j  between 0 - 1 for the sample
def soft_max_regression(j_category, x_sample, weights_matrix):
    l_categories = len(weights_matrix[0])
    max_xtw_value = 0
    for i in range(l_categories):
        max_temp = np.transpose(x_sample) @ weights_matrix[:, [i]]
        if max_temp > max_xtw_value:
            max_xtw_value = max_temp

    numerator = math.exp(np.transpose(x_sample) @ weights_matrix[:, [j_category]] - max_xtw_value)
    denominator = 0
    for i in range(l_categories):
        denominator += math.exp(np.transpose(x_sample) @ weights_matrix[:, [i]] - max_xtw_value)

    return numerator / denominator


# receives w - all weights (in matrix form?), c - hot one vectors, x - all of the samples in matrix form,
# l - number of categories/labels, m - number of samples
# returns the average error of the current weights, this we will want to minimize
def soft_max_loss(w, c, x, l, m):
    pass
    # todo implement and don't forget the trick to do element wise division u./v


# receives w - all weights (in matrix form?), c - hot one vectors, x - all of the samples in matrix form,
# m - number of samples, p - index of specific weight
# returns the gradient of weight p
def soft_max_gradient_of_weights(m, x, w, c, p):
    pass
    # understand how to use it, is it per weight?


def main():
    weights = np.asarray([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

    x_sample = np.asarray([2, 2, 2])
    j_category = 1

    print(round(soft_max_regression(j_category, x_sample, weights), 4))


    # todo 1.1
    # todo write soft_max_loss
    # todo write soft_max_gradient_of_weights
    # todo using the gradient test, make sure the derivatives are correct and submit the code
    # todo use code in UnconstrainedOPT.pdf, page 25 (translate from Julia) and produce the graph from page 24

    # todo 1.2
    # write stochastic_gradient_decent - end of week 2?
    # Using the stochastic_gradient_decent show it works on a small least squares example (??) with plots

    # todo 1.3
    # use SGD on the softmax example, with graphs after each epoch...


if __name__ == '__main__':
    main()