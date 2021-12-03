import numpy as np
import matplotlib.pyplot as plt


def f(weights, bias, x, sigmoid):
    teta_x = weights @ x
    x_plus_bias = teta_x + bias
    return sigmoid(x_plus_bias)


def derivative_b(weights, x, bias, v):
    teta_x = weights @ x
    x_plus_bias = teta_x + bias
    dev_tanh = 1 - np.power(np.tanh(x_plus_bias), 2)
    return dev_tanh * v


def derivative_w(weights, x, bias, v):
    teta_x = weights @ x
    x_plus_bias = teta_x + bias
    dev_tanh = 1 - np.power(np.tanh(x_plus_bias), 2)
    tanh_v = dev_tanh * v
    x_transpose = np.transpose(x)
    ans = tanh_v @ x_transpose
    return ans


def derivative_x(weights, x, bias, v):
    teta_x = weights @ x
    x_plus_bias = teta_x + bias
    dev_tanh = 1 - np.power(np.tanh(x_plus_bias), 2)
    tanh_v = dev_tanh * v
    transpose = np.transpose(weights)
    ans = transpose @ tanh_v
    return ans


def g(f_x, u):
    return f_x.T @ u


def the_test():
    x_sample = np.array([[1],
                         [2],
                         [3],
                         [4]])
    weights = np.random.randn(2, 4)
    bias = np.array([[1],
                     [2]])
    bias_d = np.random.randn(2, 1)

    rand_u = np.random.randn(2, 1)

    epsilon = 0.1
    y0 = []
    y1 = []
    x = []
    for i in range(0, 8):
        epsilon_i = np.power(0.5, i) * epsilon

        eps_d = (epsilon_i * bias_d)

        function_plus_eps = f(weights, bias + eps_d, x_sample, np.tanh)
        g_x_eps = g(function_plus_eps, rand_u)

        regular_function = f(weights, bias, x_sample, np.tanh)
        g_x = g(regular_function, rand_u)

        dev_b = derivative_b(weights, x_sample, bias, rand_u)
        grad_g = dev_b
        eps_d_t_grad = eps_d.T @ grad_g

        ans_1 = g_x_eps - g_x
        y0.append(abs(ans_1[0][0]))

        ans_2 = g_x_eps - g_x - eps_d_t_grad
        y1.append(abs(ans_2[0][0]))

        x.append(i)

    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.yscale('logit')
    plt.xlabel('i')
    plt.ylabel('error')
    plt.title('Successful test w.r.t bias')

    plt.show()

    # derivative_w
    weights_d = np.random.randn(2, 4)
    rand_u = np.random.randn(2, 1)

    y0 = []
    y1 = []
    x = []
    for i in range(0, 8):
        epsilon_i = np.power(0.5, i) * epsilon

        eps_d = (epsilon_i * weights_d)

        function_plus_eps = f(weights + eps_d, bias, x_sample, np.tanh)
        g_x_eps = g(function_plus_eps, rand_u)

        regular_function = f(weights, bias, x_sample, np.tanh)
        g_x = g(regular_function, rand_u)

        dev_b = derivative_w(weights, x_sample, bias, rand_u)
        grad_g = dev_b.flatten()
        flat_eps_d = eps_d.flatten()
        eps_d_t_grad = flat_eps_d.T @ grad_g

        ans_1 = g_x_eps - g_x
        y0.append(abs(ans_1[0][0]))

        ans_2 = g_x_eps - g_x - eps_d_t_grad
        y1.append(abs(ans_2[0][0]))

        x.append(i)

    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.yscale('logit')
    plt.xlabel('i')
    plt.ylabel('error')
    plt.title('Successful test w.r.t weights')

    plt.show()

    # derivative_x

    x_d = np.random.randn(4, 1)
    rand_u = np.random.randn(2, 1)

    y0 = []
    y1 = []
    x = []
    for i in range(0, 8):
        epsilon_i = np.power(0.5, i) * epsilon

        eps_d = (epsilon_i * x_d)

        function_plus_eps = f(weights, bias, x_sample + eps_d, np.tanh)
        g_x_eps = g(function_plus_eps, rand_u)

        regular_function = f(weights, bias, x_sample, np.tanh)
        g_x = g(regular_function, rand_u)

        dev_b = derivative_x(weights, x_sample, bias, rand_u)
        grad_g = dev_b
        eps_d_t_grad = eps_d.T @ grad_g

        ans_1 = g_x_eps - g_x
        y0.append(abs(ans_1[0][0]))

        ans_2 = g_x_eps - g_x - eps_d_t_grad
        y1.append(abs(ans_2[0][0]))

        x.append(i)

    plt.plot(x, y0)
    plt.plot(x, y1)
    plt.yscale('logit')
    plt.xlabel('i')
    plt.ylabel('error')
    plt.title('Successful test w.r.t x')

    plt.show()

    pass


the_test()
