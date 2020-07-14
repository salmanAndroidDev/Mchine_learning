import numpy as np

inputs = [8.5, 9.5, 9.9, 9.0]
label = 1.0
weight  = 0.5
alpha = 0.01

for i in range(100):

    pred = inputs[0] * weight
    error = (pred - label) ** 2

    delta_weight = (pred - label) * inputs[0]
    weight -= (alpha * delta_weight)
    print('Error:', error, ' Prediction: ', pred)