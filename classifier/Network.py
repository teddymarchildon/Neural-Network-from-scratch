from Layer import Layer
from Node import Node
import matrix_operations

class Network(object):
    """
    Class to represent a neural network
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        if layer is not None:
            self.layers.append(layer)

    def feed_forward(self, starting_layer_number):
        if starting_layer_number == len(self.layers):
            return
        starting_layer = self.layers[starting_layer_number - 1]
        next_layer = self.layers[starting_layer_number]
        for node in next_layer.nodes:
            input_nodes = starting_layer.nodes
            input_values = [n.value for n in input_nodes]
            weighted_sum = matrix_operations.dot_product(input_values, node.weights)
            node.forward_update(weighted_sum)
            print(node.value)
        print("done with round: ", starting_layer_number)
        self.feed_forward(starting_layer_number + 1)


if __name__ == "__main__":
    input_values = [1, 2, 3]
    network = Network()
    layer1 = Layer(3) # input layer - we don't need an activation function here
    layer1.set_values(input_values)
    
    layer2 = Layer(4, input_shape=3) # hidden layer
    layer2.set_activation_function('sigmoid')

    layer3 = Layer(2, input_shape=4)
    layer3.set_activation_function('relu')

    network.add_layer(layer1)
    network.add_layer(layer2)
    network.add_layer(layer3)

    network.feed_forward(1) # feed forward from input to layer 2
