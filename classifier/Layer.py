from Node import Node
from matrix_operations import dot_product
from activation_functions import softmax
from activation_functions import softmax_differential
from activation_functions import sigmoid
from activation_functions import sigmoid_differential
from activation_functions import relu
from activation_functions import relu_differential
from activation_functions import square_error
from activation_functions import square_error_differential

class Layer(object):
    """
    Object to represent a layer of nodes
    in the  network

    In the neural network, we apply a dot product
    of the weights with the input
    """

    def __init__(self, number_of_nodes, number_of_inputs=0):
        """
        :param number_of_nodes: type int. the number of nodes in this layer
        :param number_of_inputs: type int. The number of nodes in the previous layer
        """
        self.is_input_layer = False
        self.is_output_layer = False
        self.previous_layer = None
        self.next_layer = None
        self.set_learning_rate()
        self.nodes = []
        for _ in range(number_of_nodes):
            self.nodes.append(Node(number_of_inputs=number_of_inputs))

    def forward_update(self, node_input_values):
        """
        :param node_input_values: type list(float). The output values of the nodes in the previous layer

        This function updates the current layer with the previous layer inputs
        during feed forward step.

        It applies the activation function over the nodes in the layer.
        """
        # print("Applying %s activation function" % (self.activation_function))
        layer_input_matrix = []
        for node in self.nodes:
            input_for_node = dot_product(node_input_values, node.weights)
            layer_input_matrix.append(input_for_node)
        # We want to cache the inputs to this layer so it can be used in back propagation
        self.layer_input_matrix = layer_input_matrix

        self.__forward_update(layer_input_matrix)

    def back_propagate(self):
        """
        Apply back propagation to this layer
        """
        if self.activation_function == 'softmax':
            self.__back_propagate_softmax()
        elif self.activation_function == 'sigmoid':
            self.__back_propagate_hidden_layer(activation_function_differential=sigmoid_differential)
        elif self.activation_function == 'relu':
            self.__back_propagate_hidden_layer(activation_function_differential=relu_differential)

    def calculate_total_loss(self):
        """
        :return: type float. The total loss in the output layer, 0
            if other layer (loss should never really be 0)
        Calculate the loss for this layer
        """
        if not self.is_output_layer:
            return 0.0
        loss = 0
        for i in range(len(self.nodes)):
            if self.is_output_layer:
                loss += square_error(self.nodes[i].value, self.expected_output_values[i])
        return loss

    def set_input_values(self, values):
        """
        :param values: type list(float). The input values to the entire network

        This function sets the input values of the entire network
        """
        if not self.is_input_layer or self.is_output_layer:
            raise ValueError("Cannot set values of hidden layers. Make sure you are ",
                "setting the values of the input layer")
        if len(values) != len(self.nodes):
            raise ValueError("Make sure you have the same number of values ",
            "as nodes")
        
        for i in range(len(self.nodes)):
            self.nodes[i].value = values[i]

    def set_expected_output_values(self, output_values):
        """
        :param output_values: type list. The expected output values for this layer
        """
        if not self.is_output_layer:
            raise ValueError("Cannot set output values on a hidden layer or input layer.")
        if len(output_values) is not len(self.nodes):
            raise ValueError("Differing number of output values and nodes in the output layer.")
        self.expected_output_values = output_values

    def set_activation_function(self, function_name):
        """
        :param function_name: type str. The name of the activation function

        Sets the activation function to be used for this layer and nodes within it
        """
        self.activation_function = function_name

    def set_as_input_layer(self):
        """
        Update the appropriate instance variables for the input layer
        """
        self.is_input_layer = True
        self.is_output_layer = False

    def set_as_hidden_layer(self):
        """
        Update the appropriate instance variables for a hidden layer
        """
        self.is_input_layer = False
        self.is_output_layer = False

    def set_as_output_layer(self):
        """
        Update the appropriate instance variables for the output layer
        """
        self.is_input_layer = False
        self.is_output_layer = True

    def set_learning_rate(self, learning_rate=0.01):
        """
        :param learning_rate: type float. The learning rate for the layer

        Set the learning rate for the layer
        """
        if learning_rate < 0.001 or learning_rate > 0.1:
            raise ValueError('Learning rate should be between 0.001 and 0.1.')
        
        self.learning_rate = learning_rate

    def __forward_update(self, layer_input_matrix):
        """
        :param layer_input_matrix: type list(float). the dot product of each node's weights with the input nodes' values

        This function applies the softmax function to the nodes in the layer
        """
        layer_output_matrix = []
        for i in range(len(self.nodes)):
            self.nodes[i].forward_update(self.activation_function, layer_input_matrix, i)
            layer_output_matrix.append(self.nodes[i].value)
        self.layer_output_matrix = layer_output_matrix

    def __back_propagate_softmax(self):
        """
        The unit of the backpropagation process is the weight. Our goal
        is to minimize the loss function, so we want to investigate how
        changing a particular weight anywhere in the network will affect
        the loss function.

        We do that using the gradient of the loss function.
        """
        self.loss_differentials_wrt_activation_output = []
        self.activation_differentials_wrt_node_input = []

        for current_index in range(len(self.nodes)):
            current_node = self.nodes[current_index]

            # print(f'before back prop on node: {current_index}')
            # print(current_node.weights)

            current_activation_value = current_node.value
            loss_differential = square_error_differential(
                    current_activation_value,
                    self.expected_output_values[current_index]
                )
            activation_differential = softmax_differential(self.layer_input_matrix, current_index)
            
            self.loss_differentials_wrt_activation_output.append(loss_differential)
            self.activation_differentials_wrt_node_input.append(activation_differential)

            for i in range(len(current_node.weights)):
                '''
                These are the three components of the gradient in the output layer.

                The goal is to minimize the loss function. The loss function is a composition
                of various functions, so we use the multivariable chain rule to
                differentiate and find the gradient.
                '''
                previous_activation_value = self.previous_layer.nodes[i].value
                '''
                We then multiply all of those values together, with the learning rate,
                and update the weight for which we are considering
                '''
                total_differential = loss_differential * activation_differential * previous_activation_value
                current_node.weights[i] -= total_differential * self.learning_rate
            
            # print(f'after back prop on node {current_index}')
            # print(current_node.weights)
            # print('\n')

    def __back_propagate_hidden_layer(self, activation_function_differential):
        """
        :param activation_function_differential: type function. The differential of the 
                                                activation function in the layer

        We know we are in a hidden layer here, so we do the calculation
        a bit differently.
        """
        self.loss_differentials_wrt_activation_output = []
        self.activation_differentials_wrt_node_input = []

        for current_index in range(len(self.nodes)):
            current_node = self.nodes[current_index]
            
            # print(f'before back prop on node {current_index}')
            # print(current_node.weights)

            activation_differential = activation_function_differential(self.layer_input_matrix[current_index])

            self.activation_differentials_wrt_node_input.append(activation_differential)
            
            loss_differential = 0
            for next_node_index in range(len(self.next_layer.nodes)):
                next_layer_loss_differential = self.next_layer.loss_differentials_wrt_activation_output[next_node_index]
                next_layer_activation_differential = self.next_layer.activation_differentials_wrt_node_input[next_node_index]
                weight_between_this_node_and_that_node = self.next_layer.nodes[next_node_index].weights[current_index]
                loss_differential += (next_layer_loss_differential * next_layer_activation_differential * weight_between_this_node_and_that_node)
            
            self.loss_differentials_wrt_activation_output.append(loss_differential)

            for i in range(len(current_node.weights)):
                previous_activation_value = self.previous_layer.nodes[i].value
                total_differential = loss_differential * activation_differential * previous_activation_value
                current_node.weights[i] -= total_differential * self.learning_rate
    
            # print(f'after back prop on node {current_index}')
            # print(current_node.weights)
            # print('\n')
