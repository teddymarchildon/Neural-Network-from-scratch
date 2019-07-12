from Node import Node

class Layer(object):
    """
    Object to represent a layer of nodes
    in the  network

    In the neural network, we apply a dot product
    of the weights with the input
    """

    def __init__(self, size, input_shape=None):
        """
        :param size: the number of nodes in this layer
        :param input_shape: the shape of the input layer
        """
        self.size = size
        self.nodes = []
        self.input_shape = input_shape
        for _ in range(size):
            self.nodes.append(Node(number_of_inputs=input_shape))

    def set_values(self, values):
        if self.input_shape is not None:
            raise ValueError("Cannot set values of hidden layers. Make sure you are ",
                "setting the values of the input layer")
        if len(values) != len(self.nodes):
            raise ValueError("Make sure you have the same number of values ",
            "as nodes")
        
        for i in range(len(self.nodes)):
            self.nodes[i].value = values[i]

    def set_activation_function(self, function_name):
        for node in self.nodes:
            node.set_activation_function(function_name)
