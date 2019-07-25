from math import exp, pow

"""
A suite of functions used
in the activation function process
as well as back propagation.
"""

def sigmoid(x):
  return 1 / (1 + exp(x * -1))

def sigmoid_differential(x):
    numerator = exp(x * -1)
    denominator = pow( (1 + numerator), 2 )
    return numerator / denominator

def relu(x):
    return 0 if x < 0 else x

def relu_differential(x):
    return 0 if x <= 0 else 1

def softmax(values):
    exp_sum = sum([exp(value) for value in values])
    return [exp(value)/exp_sum for value in values]

def softmax_differential(values, index):
    exp_sum = sum([exp(value) for value in values])
    numerator_exp = exp(values[index])
    softmax_at_index = numerator_exp / exp_sum
    return softmax_at_index * (1 - softmax_at_index)

def square_error(a, b):
    return pow( (a - b), 2 )

def square_error_differential(a, b):
    return 2 * (a - b)
