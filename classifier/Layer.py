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

    def __init__(self, size, input_shape=None):
        """
        :param size: type int. the number of nodes in this layer
        :param input_shape: type int. the shape of the input layer
        TODO Change input shape to accept a list so that we can put in a ton of samples at once
        """
        self.size = size
        self.nodes = []
        self.input_shape = input_shape
        self.is_input_layer = False
        self.is_output_layer = False
        self.previous_layer = None
        self.next_layer = None
        for _ in range(size):
            self.nodes.append(Node(number_of_inputs=input_shape))

    def set_as_input_layer(self):
        self.is_input_layer = True
        self.is_output_layer = False

    def set_as_hidden_layer(self):
        self.is_input_layer = False
        self.is_output_layer = False

    def set_as_output_layer(self):
        self.is_input_layer = False
        self.is_output_layer = True

    def forward_update(self, node_input_values):
        """
        :param node_input_values: type list(float). the output values of the nodes in the previous layer
        
        This function updates the current layer with the previous layer inputs
        during feed forward step.

        It applies the activation function over the nodes in the layer.
        """
        print("Applying %s activation function" % (self.activation_function))
        layer_input_matrix = []
        for node in self.nodes:
            input_for_node = dot_product(node_input_values, node.weights)
            layer_input_matrix.append(input_for_node)
        # We want to cache the inputs to this layer so it can be used in back propagation
        self.layer_input_matrix = layer_input_matrix

        self.__forward_update(layer_input_matrix)

    def back_propagate(self):
        """
        """
        if self.activation_function == 'softmax':
            self.__back_propagate_softmax()
        elif self.activation_function == 'sigmoid':
            self.__back_propagate_sigmoid()
        elif self.activation_function == 'relu':
            self.__back_propagate_relu()

    def set_input_values(self, values):
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
    
    def calculate_total_loss(self):
        """
        Calculate the loss for this layer
        """
        if not self.is_output_layer:
            return 0.0
        loss = 0
        for i in range(len(self.nodes)):
            if self.is_output_layer:
                loss += square_error(self.nodes[i].value, self.expected_output_values[i])
        return loss

    def __forward_update(self, layer_input_matrix):
        """
        :param weighted_sums: type list(float). the dot product of each node's weights with the input nodes' values

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
        current_activation_outputs = [node.value for node in self.nodes]
        for current_index in range(len(self.nodes)):
            current_node = self.nodes[current_index]
            for i in range(len(current_node.weights)):
                current_activation_value = current_node.value
                '''
                These are the three components of the gradient in the output layer.

                The goal is to minimize the loss function. The loss function is a composition
                of various functions, so we use the multivariable chain rule to
                differentiate and find the gradient.
                '''
                square_error_loss_differential = square_error_differential(current_activation_value, self.expected_output_values[current_index])
                activation_differential = softmax_differential(current_activation_outputs, i)
                previous_activation_value = self.previous_layer.nodes[i].value

                '''
                We then multiply all of those values together, with the learning rate,
                and update the weight for which we are considering
                '''
                total_differential = square_error_loss_differential * activation_differential * previous_activation_value
                current_node.weights[i] = total_differential * 0.001

    def __back_propagate_sigmoid(self):
        """
        """
        for current_index in range(len(self.nodes)):
            current_node = self.nodes[current_index]
            for i in range(len(current_node.weights)):
                current_activation_value = current_node.value

                # TODO understand where the differential of the loss function plays into hidden layers
                square_error_loss_differential = square_error_differential(current_activation_value, 5)
                # TODO Fix this to accept the input to the node, not activation output
                activation_differential = sigmoid_differential(current_node.value)

                previous_activation_value = self.previous_layer.nodes[i].value

                total_differential = square_error_loss_differential * activation_differential * previous_activation_value
                current_node.weights[i] = total_differential * 0.01
    
    def __back_propagate_relu(self):
        """
        """
        for current_index in range(len(self.nodes)):
            current_node = self.nodes[current_index]
            for i in range(len(current_node.weights)):
                current_activation_value = current_node.value

                # TODO understand where the differential of the loss function plays into hidden layers
                square_error_loss_differential = square_error_differential(current_activation_value, 5)
                # TODO Fix this to accept the input to the node, not activation output
                activation_differential = relu_differential(current_node.value)

                previous_activation_value = self.previous_layer.nodes[i].value

                total_differential = square_error_loss_differential * activation_differential * previous_activation_value
                current_node.weights[i] = total_differential * 0.01
