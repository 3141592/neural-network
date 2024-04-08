import numpy

class neuralNetwork:
    # init neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.ly = learningrate

        # link weights
        self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.woh = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)

        pass

    def train():
        pass

    def query():
        pass

input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

