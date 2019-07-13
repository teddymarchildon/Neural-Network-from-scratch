from random import uniform
from activation_functions import sigmoid, relu

class Node(object):
    """
    Object to represent a node in a neural network
    """

    def __init__(self, number_of_inputs=None):
        """
        :param number_of_inputs: type int. the number of inputs into this node
        
        The weights here are represented as an array, where the value at
        i is the weight between this node, and the ith previous node
        """
        self.value = 0
        if number_of_inputs is None:
            self.weights = None
        else:
            self.weights = [uniform(0, number_of_inputs) for _ in range(number_of_inputs)]

    def forward_update(self, weighted_input):
        """
        :param weighted_input: type int. the dot product of this node's weights with the previous layer's node values

        Updates the value of this node with the dot product, and subsequently the activated value
        """
        self.value = weighted_input
        self.__apply_activation_function()

    def set_activation_function(self, activation_function):
        """
        :param activation_function: type str. The name of the activation function to be used for the node
        """
        self.activation_function = activation_function

    def __apply_activation_function(self):
        """
        Applies the activation function to the value in the node
        """
        if self.activation_function is None:
            raise ValueError('Please specify an activation function')
        if self.activation_function == 'sigmoid':
            self.value = sigmoid(self.value)
        elif self.activation_function == 'relu':
            self.value = relu(self.value)
        else:
            raise ValueError('Activation Function not found')
