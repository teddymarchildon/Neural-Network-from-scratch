from random import gauss
from math import sqrt
from activation_functions import sigmoid, relu, softmax

class Node(object):
    """
    Object to represent a node in a neural network
    """

    def __init__(self, number_of_inputs):
        """
        :param number_of_inputs: type int. the number of inputs into this node. 
                                0 if it's in the input layer.
        
        The weights here are represented as an array, where the value at
        i is the weight between this node, and the ith previous node

        We are going to use Xavier initialization in order to reduce the 
        unstable gradient issues that may arise. This centers the variance
        of the weights coming into this node at 1 / number of inputs to 
        the node.

        The weight is random value pulled from a Normal(0, 1) distribution.
        """
        self.value = 0
        if number_of_inputs > 0:
            xavier_coefficient = sqrt(1 / number_of_inputs)
            self.weights = [gauss(0, 1)*xavier_coefficient for _ in range(number_of_inputs)]
        else: # No weights coming into a node in the input layer
            self.weights = []

    def forward_update(self, activation_function, layer_input_matrix, index):
        """
        :param weighted_input: type int. the dot product of this node's weights with the previous layer's node values

        Updates the value of this node with the dot product, and subsequently the activated value
        """
        self.value = layer_input_matrix[index]
        self.__apply_activation_function(activation_function, layer_input_matrix, index)

    def __apply_activation_function(self, activation_function, layer_input_matrix, index):
        """
        :param layer_input_matrix: type list. The input values to the nodes in this layer,
            the weighted sum of the previous layer's nodes and the weights in each node in this layer
        :param index: The index of this node in the layer matrix

        Applies the activation function to the value in the node
        """
        if activation_function == 'sigmoid':
            self.value = sigmoid(self.value)
        elif activation_function == 'relu':
            self.value = relu(self.value)
        elif activation_function == 'softmax':
            self.value = softmax(layer_input_matrix)[index]
        else:
            raise ValueError('Activation Function not found')
