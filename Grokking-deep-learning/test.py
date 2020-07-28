import numpy as np

input = np.array([[4.0]])
weight_0_1 = np.array([[1.0]])
weight_1_2 = np.array([[0.5]])
output = np.array([3.0])
alpha = 0.1


for iter in range(20):
    hidden = input.dot(weight_0_1)
    pred = hidden.dot(weight_1_2)

    error = (pred - output) ** 2
    delta = (pred - output)

    hidden_delta = delta * weight_1_2

    weight_1_2 -= alpha * (hidden * delta)
    weight_0_1 -= alpha * hidden_delta

    print(error,' weight 1:', weight_0_1,' weight 2:', weight_1_2,' prediction: ', pred, 'hidden', hidden)
