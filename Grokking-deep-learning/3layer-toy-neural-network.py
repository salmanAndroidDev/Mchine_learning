import numpy as np

np.random.seed(1)

def relu(x):
    return (x > 0) * x

def delta_remover(x):
    return x > 0

streetlights = np.array( [[ 1, 0, 1 ],
                          [ 0, 1, 1 ],
                          [ 0, 0, 1 ],
                          [ 1, 1, 1 ] ] )

walk_vs_stop = np.array([[ 1, 1, 0, 0]]).T

alpha = 0.2
hidden_size = 4

layer_1_weight = 2 * np.random.random((3, hidden_size)) -1
layer_2_weight = 2 * np.random.random((hidden_size,1)) - 1

for iteration in range(60):
    total_error = 0
    for i in range(len(streetlights)):

        input_layer = streetlights[i:i+1]    
        hidden_layer = relu(np.dot(input_layer,layer_1_weight))
        output_layer = np.dot(hidden_layer, layer_2_weight)

        total_error += np.sum((output_layer - walk_vs_stop[i:i+1]) ** 2)
        
        output_delta = output_layer - walk_vs_stop[i:i+1]
        hidden_delta = output_delta.dot(layer_2_weight.T) * delta_remover(hidden_layer)

        layer_2_weight -= alpha * (hidden_layer.T.dot(output_delta))
        layer_1_weight -= alpha * (input_layer.T.dot(hidden_delta))
        
    if (iteration % 10 == 9) :
        print(total_error)
