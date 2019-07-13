from math import exp

def sigmoid(x):
  return 1 / (1 + exp(-x))

def relu(x):
    return 0 if x < 0 else x

def softmax(values):
    exp_sum = sum([exp(value) for value in values])
    return [exp(value)/exp_sum for value in values]
