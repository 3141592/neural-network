import numpy
import scipy.special

class neuralNetwork:
    # init neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.ly = learningrate

        # link weights
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train():
        pass

    def query(self, inputs_list):
        # convert inputs_list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate signals emergin from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# test
final_outputs = n.query([1.0, 0.5, -1.5])
print("final_outputs: ", final_outputs)


