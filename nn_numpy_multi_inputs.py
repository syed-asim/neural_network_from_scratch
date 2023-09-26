import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]
          ]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output_layer = []
# for i, w in enumerate(weights):
#     output_layer.append(np.dot(inputs, w)+biases[i])
# print(output_layer)
wXi = np.dot(inputs, np.array(weights).T)
output_layer = wXi+biases
print(output_layer)
