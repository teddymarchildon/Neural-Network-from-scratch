from Node import Node
from matrix_operations import dot_product
from activation_functions import softmax

class Layer(object):
    """
    Object to represent a layer of nodes
    in the  network

    In the neural network, we apply a dot product
    of the weights with the input
    """

    def __init__(self, size, input_shape=None):
        """
        :param size: type int. the number of nodes in this layer
        :param input_shape: type int. the shape of the input layer
        TODO Change input shape to accept a list so that we can put in a ton of samples at once
        """
        self.size = size
        self.nodes = []
        self.input_shape = input_shape
        for _ in range(size):
            self.nodes.append(Node(number_of_inputs=input_shape))

    def forward_update(self, node_input_values):
        """
        :param node_input_values: type list(float). the values of the previous layer nodes
        
        This function updates the current layer with the previous layer inputs
        during feed forward step.

        It applies the activation function over the nodes in the layer.
        """
        print("Applying %s activation function" % (self.activation_function))
        if self.activation_function == 'softmax': # Softmax requires knowledge of the nodes, so we appy it here
            weighted_sums = []
            for node in self.nodes:
                weighted_sums.append(dot_product(node_input_values, node.weights))
            self.__forward_update_softmax(weighted_sums)
        else: # Other activation functions only pertain to the individual node, so we can update within the node
            for node in self.nodes:
                weighted_sum = dot_product(node_input_values, node.weights)
                node.forward_update(weighted_sum)

    def __forward_update_softmax(self, weighted_sums):
        """
        :param weighted_sums: type list(float). the dot product of each node's weights with the input nodes' values

        This function applies the softmax function to the nodes in the layer
        """
        updated_values = softmax(weighted_sums)
        for i in range(len(weighted_sums)):
            node = self.nodes[i]
            node.value = updated_values[i]

    def set_values(self, values):
        """
        :param values: type list(float). The input values to the entire network

        This function sets the input values of the entire network
        """
        if self.input_shape is not None:
            raise ValueError("Cannot set values of hidden layers. Make sure you are ",
                "setting the values of the input layer")
        if len(values) != len(self.nodes):
            raise ValueError("Make sure you have the same number of values ",
            "as nodes")
        
        for i in range(len(self.nodes)):
            self.nodes[i].value = values[i]

    def set_activation_function(self, function_name):
        """
        :param function_name: type str. The name of the activation function

        Sets the activation function to be used for this layer and nodes within it
        """
        self.activation_function = function_name
        for node in self.nodes:
            node.set_activation_function(function_name)
