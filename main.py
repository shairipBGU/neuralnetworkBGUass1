import math
import numpy as np


# receives weights_matrix - all weights (in matrix form?), j_category - the label we want to check zero based,
# x_sample - the sample, l_categories_size - number of categories
# returns a probability of how much the classifier is sure the x_sample label(y) is j_category  between 0 - 1 for the sample
def soft_max_regression(j_category, x_sample, weights_matrix, l_categories_size):
    max_xtw_value = 0
    for i in range(l_categories_size):
        max_temp = np.transpose(x_sample) @ weights_matrix[:, [i]]
        if max_temp > max_xtw_value:
            max_xtw_value = max_temp

    numerator = math.exp(np.transpose(x_sample) @ weights_matrix[:, [j_category]] - max_xtw_value)
    denominator = 0
    for i in range(l_categories_size):
        denominator += math.exp(np.transpose(x_sample) @ weights_matrix[:, [i]] - max_xtw_value)

    return numerator / denominator


# receives weights_matrix - all weights (in matrix form),
# class_hot_one_vector_matrix - hot one vectors of class for each sample,
# x_all_data - all of the samples in matrix form, l_categories_size - number of categories/labels,
# m_sample_size - number of samples
# returns the average error of the current weights, this we will want to minimize
def soft_max_loss(weights_matrix, class_hot_one_vector_matrix, x_all_data, l_categories_size, m_sample_size):
    x_t = np.transpose(x_all_data)
    error_sum = 0
    for k in range(l_categories_size):
        # todo probably need to do safe exp calculation here too, (but what is a maximum on a vector?)
        numerator = np.exp(x_t @ weights_matrix[:, [k]])

        denominator = np.exp(x_t @ weights_matrix[:, [0]])
        for j in range(1, l_categories_size):
            denominator += np.exp(x_t @ weights_matrix[:, [j]])

        divide_result = numerator / denominator
        inner_parentheses = np.log(divide_result)
        error_sum += np.transpose(class_hot_one_vector_matrix[k]) @ inner_parentheses

    return -1 * error_sum / m_sample_size


# receives weights_matrix - all weights (in matrix form),
# class_hot_one_vector_matrix - hot one vectors of class for each sample,
# x_all_data - all of the samples in matrix form, l_categories_size - number of categories/labels,
# m_sample_size - number of samples, p - index of specific weight
# returns the gradient of weight p
def soft_max_gradient_of_weight_p(weights_matrix, class_hot_one_vector_matrix, x_all_data, l_categories_size,
                                  m_sample_size, p_index_of_weight):
    x_t = np.transpose(x_all_data)
    numerator = np.exp(x_t @ weights_matrix[:, [p_index_of_weight]])

    denominator = np.exp(x_t @ weights_matrix[:, [0]])
    for j in range(1, l_categories_size):
        denominator += np.exp(x_t @ weights_matrix[:, [j]])

    divide_result = numerator / denominator
    # todo not sure about the transpose but let's see... suppose to be the indicator vector
    class_row = class_hot_one_vector_matrix[p_index_of_weight]
    class_p = np.transpose(np.atleast_2d(class_row))
    inner_parentheses = divide_result - class_p
    product = x_all_data @ inner_parentheses
    return product / m_sample_size



def main():
    weights = np.asarray([[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]])

    x_sample = np.asarray([2, 2, 2])
    j_category = 2

    # print(round(soft_max_regression(j_category, x_sample, weights, 3), 4))

    class_hot_one_vector_matrix = np.eye(3)
    x_all_data = np.asarray([[1, 2, 3],
                             [1, 2, 3],
                             [1, 2, 3]])
    l_categories_size = 3
    m_sample_size = 3
    # print(soft_max_loss(weights, class_hot_one_vector_matrix, x_all_data, l_categories_size, m_sample_size))

    p_index_of_weight = 0
    print(soft_max_gradient_of_weight_p(weights, class_hot_one_vector_matrix, x_all_data, l_categories_size, m_sample_size, p_index_of_weight))
    print(soft_max_gradient_of_weight_p(weights, class_hot_one_vector_matrix, x_all_data, l_categories_size, m_sample_size, 1))
    print(soft_max_gradient_of_weight_p(weights, class_hot_one_vector_matrix, x_all_data, l_categories_size, m_sample_size, 2))


    # todo 1.1
    # todo write soft_max_loss - don't forget to add bias to the weights
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