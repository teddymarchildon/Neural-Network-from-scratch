import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def relu(x):
    return 0 if x < 0 else x
