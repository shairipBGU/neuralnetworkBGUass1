# receives w - all weights (in matrix form?), j - the label we want to check, x - the sample,
# l - number of categories/labels
# returns a probability of how much the classifier is sure the X sample label(y) is j  between 0 - 1 for the sample
def soft_max_regression(j, x, w, l):
    # todo implement and don't forget the trick to cancel out the infinity in exp
    pass


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
    pass


if __name__ == '__main__':
    main()