from Layer import Layer
from Node import Node
import matrix_operations

class Network(object):
    """
    Class to represent a neural network
    """

    def __init__(self):
        """
        Create the Network object
        """
        self.layers = []
        self.expected_output = []
        self.learning_rate = 0.001

    def add_layer(self, layer):
        """
        :param layer: type Layer. A layer to be added to the network
        """
        if layer is None or not isinstance(layer, Layer):
            raise ValueError('Make sure you create a type Layer before adding it to the network')
        if len(self.layers) is 0:
            layer.set_as_input_layer()
            self.layers.append(layer)
            return
        if len(self.layers) is 1:
            layer.set_as_output_layer()
            # Set the input layer's next layer
            self.layers[len(self.layers) - 1].next_layer = layer
            # Set this layer's previous layer
            layer.previous_layer = self.layers[len(self.layers) - 1]
            self.layers.append(layer)
            return
        # Update the previous layers status as a hidden layer
        self.layers[len(self.layers) - 1].set_as_hidden_layer()
        # Set the previous layer's next layer as this layer
        self.layers[len(self.layers) - 1].next_layer = layer
        # Set this layer's previous layer as the previous layer
        layer.previous_layer = self.layers[len(self.layers) - 1]
        # Set this layer as the new output layer
        layer.set_as_output_layer()
        self.layers.append(layer)

    def set_input_values(self, input_values):
        """
        :param input_values: type list. The input values for the network

        TODO calculate the number of training samples here to be used in backprop
        """
        if input_values is None or len(input_values) == 0:
            raise ValueError('Please ensure the input values are populated.')
        if type(input_values[0]) is not list:
            raise ValueError('Please use a double array for your input values.')

        self.num_samples = len(input_values)
        self.num_input_nodes = len(input_values[0])
        
        self.network_input = input_values
        # Process the first set of data
        self.layers[0].set_input_values(input_values[0])

    def set_expected_output_values(self, expected_output_values):
        """
        :param expected_output_values: type list. The expected output values of the network
        """
        if expected_output_values is None or len(expected_output_values) == 0:
            raise ValueError('Please ensure the output values are populated.')
        if type(expected_output_values[0]) is not list:
            raise ValueError('Please use a double array for your output values.')
        
        self.layers[len(self.layers) - 1].set_expected_output_values(expected_output_values[0])

    def set_learning_rate(self, learning_rate=0.01):
        """
        :param learning_rate: type float. The learning rate for the network
        """
        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

    def feed_forward(self):
        """
        External function for feed forward
        """
        self.__feed_forward_recursively(starting_layer_number=1)

    def back_propagate(self):
        """
        Function for back propagating the neural network
        """
        # print('Beginning backpropagation')
        self.__back_propagate_recursive(starting_layer_number=len(self.layers))

    def predict(self, test_data):
        """
        :param test_data: type list. Function for predicting output on test data
        """
        pass

    def __feed_forward_recursively(self, starting_layer_number):
        """
        :param starting_layer_number: type int. The starting layer for 
                        feed forward. Recursively updated.
        """
        if starting_layer_number == len(self.layers):
            loss = self.layers[len(self.layers) - 1].calculate_total_loss()
            print(f'Loss at output: {loss}')
            return

        # We will never do any forward updating on the input layer
        # because it only occurs on the next layer
        starting_layer = self.layers[starting_layer_number - 1]
        initial_node_values = [n.value for n in starting_layer.nodes]

        next_layer = self.layers[starting_layer_number]
        next_layer.forward_update(initial_node_values)

        # print([node.value for node in next_layer.nodes])
        # print(f'done with layer: {starting_layer_number}')

        self.__feed_forward_recursively(starting_layer_number + 1)
    
    def __back_propagate_recursive(self, starting_layer_number):
        """
        :param starting_layer_number: type int. The layer to start back propagation

        Recursively applies back propagation through the network until the input layer.
        """
        if starting_layer_number == 1:
            # print('Reached input layer; done back propagating')
            return
        current_layer = self.layers[starting_layer_number - 1]
        current_layer.back_propagate()
        # print(f'done with layer: {starting_layer_number}\n')
        self.__back_propagate_recursive(starting_layer_number - 1)


if __name__ == "__main__":
    input_values = [[2, 4, 6, 8, 10, 12, 14, 15]]
    output_values = [[1, 0, 0]]

    for _ in range(250):
        input_values[0].append(5)

    network = Network()

    # input layer - we don't need an activation function here
    layer1 = Layer(number_of_nodes=len(input_values[0]))

    layer2 = Layer(number_of_nodes=4, number_of_inputs=len(input_values[0])) # hidden layer
    layer2.set_activation_function('relu')

    layer3 = Layer(number_of_nodes=2, number_of_inputs=4)
    layer3.set_activation_function('sigmoid')

    # The output layer should use softmax for conversion to probabilities
    layer4 = Layer(number_of_nodes=len(output_values[0]), number_of_inputs=2)
    layer4.set_activation_function('softmax')

    network.add_layer(layer1)
    network.add_layer(layer2)
    network.add_layer(layer3)
    network.add_layer(layer4)

    network.set_learning_rate(0.01)

    network.set_input_values(input_values)
    network.set_expected_output_values(output_values)

    for i in range(50):
        network.feed_forward() # feed forward from input to layer 2
        # print('\n')
        network.back_propagate()
        # print(f'Done with round: {i}')
