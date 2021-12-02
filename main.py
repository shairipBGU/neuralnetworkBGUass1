import numpy as np
import matplotlib.pyplot as plot
import random

# receives weights_matrix - all weights (in matrix form)
# returns a probability of how much the classifier is sure the x_sample label(y)
# is j_category  between 0 - 1 for the sample
def soft_max_func(weights_matrix):
    numerator = np.exp(weights_matrix)
    return np.divide(numerator, np.sum(numerator, axis=0, keepdims=True))

# receives weights_matrix - all weights (in matrix form),
# class_hot_one_vector_matrix - hot one vectors of class for each sample,
# m_sample_size - number of samples
# returns the average error of the current weights, this we will want to minimize
def soft_max_loss(weights_matrix, class_hot_one_vector_matrix, m_sample_size):
    inner_log_calc = np.log(soft_max_func(weights_matrix))
    error_sum = np.sum(class_hot_one_vector_matrix * inner_log_calc, keepdims=True)
    return -1 * error_sum / m_sample_size

# here we want to return the the gradiant of the wieghts matrix respect to b and w
# we take the wight matrix  and teh number of sampels and teh vez and teh hot vec
#we return a list were we calculate respect to w and respect to b
#the returned list[0] its respect to be and the returned list[1] si respect to w
def grad_func(weights_matrix, vec, m_sample_size, class_hot_one_vector_matrix):
    cons = np.divide(1, m_sample_size)
    prod = (soft_max_func(weights_matrix) - class_hot_one_vector_matrix)
    res_respect_to_b = cons * prod
    res_respect_to_w = cons*(np.dot(vec, prod.T))
    res_list = [res_respect_to_b, res_respect_to_w]
    return res_list

def yacob_grad_test_func():
    print(" need to be done, understand what they mean and meet burak for it")

# we dont want to get a 11 or 00 vec we ned 01 or 10, this will correct if there is a mistake.
def corrector(c, len):
    print(c)
    for i in range(0, len):
        if c[:, i][0] == 0.0:
            c[:, i][1] = 1.0
        else:
            c[:, i][1] = 0.0


    return c

def sgd_func():
    training_set, validation_set, a, l,counter  = 1000, 200, 4, 2,0
    total_set = training_set + validation_set
    W = np.random.randn(a, l)
    b = np.random.randn(l, 1)
    dataset = np.random.randn(a, total_set)

    training_dat = dataset[:, 0:training_set]
    training_class = np.random.choice([0, 1], size=(l, training_set))
    training_class = corrector(training_class,len(training_class.T))
    validation_dat = dataset[:, training_set:total_set]
    val_cl = np.random.choice([0, 1], size=(l, validation_set))
    val_cl = corrector(val_cl,len(val_cl.T))

    res_x, res_y_succ_train, res_y__succ_validate, learning_rate= [], [], [], 0.001
    for count in range(0, 100):
        training_success,validation_success = 0,0
        for dat in range(0, len(training_dat[0])):
            the_dat,class_dat = training_dat[:, dat].reshape(a, 1), training_class[:, dat].reshape(l, 1)
            S = soft_max_func(np.dot(W.T, the_dat) + b)
            if(S[0] >= 0.5):
                S[0] = 1
            else:
                S[0] = 0
            if (S[1] < 0.5):
                S[1] = 0
            else:
                S[1] = 1
            if S[0] == class_dat[0][0]:
                training_success += 1
            training_set = 1
            b = b - learning_rate * grad_func(np.dot(W.T, the_dat) + b,the_dat ,training_set, class_dat)[0]
            W = W - learning_rate * grad_func(np.dot(W.T, the_dat) + b, the_dat, training_set, class_dat)[1]
        for dat in range(0, len(validation_dat[0])):
            the_dat = validation_dat[:, dat].reshape(a, 1)
            class_dat = val_cl[:, dat].reshape(l, 1)
            S_val = soft_max_func(np.dot(W.T, the_dat) + b)
            if(S_val[0] >= 0.5):
                S_val[0] = 1
            else:
                S_val[0] = 0

            if (S_val[1] < 0.5):
                S_val[1] = 0
            else:
                S_val[1] = 1
            if S_val[0] == class_dat[0][0]:
                validation_success += 1

        t_suc =  np.divide(training_success,len(training_dat[0]))
        res_y_succ_train = res_y_succ_train + [t_suc]
        v_suc = np.divide(validation_success,len(validation_dat[0]))
        res_y__succ_validate = res_y__succ_validate + [v_suc]
        res_x = res_x + [count]
    plot.xlabel('Epoc')
    plot.ylabel('Success Rate')
    plot.title('sgd_func (learning rate=0.001)')
    plot.plot(res_x, res_y_succ_train,  label='Training')
    plot.plot(res_x, res_y__succ_validate,  label='Validation')
    plot.legend()
    plot.show()

# def f(x):  # the equivlant in julia:  F = x -> 0.5*dot(x,x) -> means return / dot(x,x) this compute steh product betwen
#     # two vectors and returns a scalar. https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/
#     return 0.5 * (x @ x)
#
# def g_f(x):  # g_F = x->x
#     return x
#
# def gradiant_test_ver():
#     n = 20
#     x = np.random.randn(n)
#     d = np.random.randn(n)
#     epsilon = 0.1
#     f0 = f(x)
#     g0 = g_f(x)
#     y0 = np.zeros(8)
#     y1 = np.zeros(8)
#     print(("k\terror order 1 \t\t error order 2"))
#     for k in range(1, 8):
#         epsk = epsilon * (0.5 ** k)
#         fk = f(x + epsk * d)
#         f1 = f0 + epsk * (g0 @ d)
#         y0[k] = abs(fk - f0)
#         y1[k] = abs(fk - f1)
#         print(k, "\t", abs(fk - f0), "\t", abs(fk - f1))
#
#     plot.semilogy(np.arange(1, 9, 1), y0)  # collect(1:8) =  np.arange(1, 8, 1)
#     plot.semilogy(np.arange(1, 9, 1), y1)
#
#     plot.legend(("Zero order approx", "First order approx"))
#     plot.title("Successful Grad test in semilogarithmic plot")
#     plot.xlabel("k")
#     plot.ylabel('error')
#     plot.show()


def main():
    print("ok")
    sgd_func()

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
