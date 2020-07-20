import numpy as np

matrix = [
                    [1,1,1],
                    [3,2,1],
                    [2,1,2]
        ]
price = [15,28,23]


inv = np.linalg.solve(matrix, price)
print(inv)