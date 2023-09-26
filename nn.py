# Input to layer 1 (single Neuron)
# inputs = [1, 2, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2
# output = inputs[0]*weights[0] + inputs[1] * \
#     weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3]+bias
# print(output)

# Input to layer 1 (single Neuron)
inputs = [1, 2, 3, 2.5]

weight1 = [0.2, 0.8, -0.5, 1.0]
weight2 = [0.5, -0.91, 0.26, -0.5]
weight3 = [-0.26, -0.27, 0.17, 0.87]
weights = [weight1, weight2, weight3]
bias1 = 2
bias2 = 3
bias3 = 0.5
biases = [bias1, bias2, bias3]

# Classic code
# layer_outputs = []
# for i in range(len(biases)):
#     neuron_val = 0.0
#     for j in range(len(inputs)):
#         neuron_val += weights[i][j]*inputs[j]
#     neuron_val += biases[i]
#     layer_outputs.append(neuron_val)
# print(layer_outputs)

# Modren way
layer_outputs = []
for bias, ws in zip(biases, weights):
    neuron_val = 0
    for inp, w in zip(inputs, ws):
        neuron_val += inp*w
    neuron_val += bias
    layer_outputs.append(neuron_val)
print(layer_outputs)
