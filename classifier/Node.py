from random import uniform
import activation_functions

class Node(object):
    """
    Object to represent a node in a neural network
    """

    def __init__(self, number_of_inputs=None):
        """
        :param number_of_inputs: the number of inputs into this node
        
        The weights here are represented as an array, where the value at
        i is the weight between this node, and the ith previous node
        """
        self.value = 0
        if number_of_inputs is None:
            self.weights = None
        else:
            self.weights = [uniform(0, number_of_inputs) for _ in range(number_of_inputs)]

    def forward_update(self, weighted_input):
        self.value = weighted_input
        self.apply_activation_function()

    def set_activation_function(self, activation_function):
        self.activation_function = activation_function

    def apply_activation_function(self):
        print("Applying %s activation function" % (self.activation_function))
        if self.activation_function is None:
            raise ValueError('Please specify an activation function')
        if self.activation_function == 'sigmoid':
            self.value = activation_functions.sigmoid(self.value)
        elif self.activation_function == 'relu':
            self.value = activation_functions.relu(self.value)
        else:
            raise ValueError('Activation Function not found')

