import numpy
import matplotlib.pyplot
import scipy.special

class neuralNetwork:
    # init neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # link weights
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # activation function
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
 
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output error split by weights
        # recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def query(self, inputs_list):
        # convert inputs_list to 2D array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = numpy.dot(self.wih, inputs)

        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # calculate the signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# test
#final_outputs = n.query([1.0, 0.5, -1.5])
#print("final_outputs: ", final_outputs)

# load the mnist training data CSV file into a list
training_data_file = open("data/mnist_test_100.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

all_values_test = training_data_list[0].split(',')
image_array = numpy.asfarray(all_values_test[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')
#matplotlib.pyplot.show()

scaled_input = (numpy.asfarray(all_values_test[1:]) / 255.0 * 0.99) + 0.01
print(scaled_input)

# output nodes
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values_test[0])] = 0.99
print("targets: ")
print(targets)

#
# train the neural network
# go through all records in the training data set
for record in training_data_list:
    # split the record by the commas
    all_values = record.split(',')
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # create the target output values
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)

# load the mnist test data CSV file into a list
test_data_file = open("data/mnist_test_10.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# get the first test record
all_values = test_data_list[1].split(',')
# print the label
print(all_values[0])
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')
matplotlib.pyplot.show()

query = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
print("query: ")
print(query)







