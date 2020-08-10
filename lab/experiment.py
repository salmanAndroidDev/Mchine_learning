import numpy as np
# 0 = salman
# 1 = ali
# 2 = vahab

P = np.array([ # all the probabilities that we belive have happened.
    [0.2, 0.3, 0.1]
])

def cross_entropy():
    result = []
    Y = np.array([
        [1,0,1],
        [0,1,0],
    ])
    sum = 0
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            sum += Y[i][j] * -np.log(P[i][j])
    print(sum)

result = cross_entropy()
print(result)