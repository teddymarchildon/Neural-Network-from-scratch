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

    def add_layer(self, layer):
        """
        :param layer: type Layer. A layer to be added to the network
        """
        if layer is None:
            raise ValueError("Make sure you create a layer before adding it to the network")
        self.layers.append(layer)

    def feed_forward(self, starting_layer_number):
        """
        :param starting_layer_number: type int. The starting layer for 
                        feed forward. Recursively updated.
        """
        if starting_layer_number < 1 or starting_layer_number > len(self.layers):
            raise ValueError("Make sure the starting layer is within the network")

        if starting_layer_number == len(self.layers):
            return

        # We will never do any forward updating on the input layer
        # because it only occurs on the next layer
        starting_layer = self.layers[starting_layer_number - 1]
        input_values = [n.value for n in starting_layer.nodes]

        next_layer = self.layers[starting_layer_number]
        next_layer.forward_update(input_values)

        print([node.value for node in next_layer.nodes])
        print("done with layer: ", starting_layer_number)

        self.feed_forward(starting_layer_number + 1)


if __name__ == "__main__":
    input_values = [2, 4, 6]
    network = Network()
    layer1 = Layer(3) # input layer - we don't need an activation function here
    layer1.set_values(input_values)
    
    layer2 = Layer(4, input_shape=3) # hidden layer
    layer2.set_activation_function('relu')

    layer3 = Layer(2, input_shape=4)
    layer3.set_activation_function('sigmoid')

    layer4 = Layer(2, input_shape=2)
    layer4.set_activation_function('softmax')

    network.add_layer(layer1)
    network.add_layer(layer2)
    network.add_layer(layer3)
    network.add_layer(layer4)

    network.feed_forward(1) # feed forward from input to layer 2
