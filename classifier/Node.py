from random import uniform
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
        """
        self.value = 0
        # randomly initialize the weights
        self.weights = [uniform(0, number_of_inputs) for _ in range(number_of_inputs)]

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
