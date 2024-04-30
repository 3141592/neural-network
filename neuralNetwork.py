import numpy
import matplotlib.pyplot
import scipy.special
import imageio.v3
import glob

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

    def train(self, inputs_list, targets_list, record, cummulative_average):
        # Logging
        # Epoch 20/20
        # 30/30 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9972 - loss: 0.0205 - val_accuracy: 0.8677 - val_loss: 0.5793

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
        cummulative_average = (cummulative_average + hidden_errors.mean()) / 2.0
        #print("    Record:    Error: ", hidden_errors.sum(), end='\r')
        print("    Record: {} Error: {}".format(record, cummulative_average), end='\r')

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        return cummulative_average

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
hidden_nodes = 125
output_nodes = 10
learning_rate = 0.175

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


print("# load the mnist training data CSV file into a list")
training_data_file = open("../data/mnist/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

all_values_test = training_data_list[0].split(',')
image_array = numpy.asfarray(all_values_test[1:]).reshape((28,28))

scaled_input = (numpy.asfarray(all_values_test[1:]) / 255.0 * 0.99) + 0.01

# output nodes
onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values_test[0])] = 0.99

#
# train the neural network

# epochs is the number of times the training data is used for training
epochs = 5

print("")
print("# go through all records in the training data set")
for e in range(epochs):
    # Logging
    # Epoch 20/20
    # 30/30 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.9972 - loss: 0.0205 - val_accuracy: 0.8677 - val_loss: 0.5793
    print("Starting Epoch {}/{}.".format(e + 1,epochs))
    count = 0
    cummulative_average = 0
    for record in training_data_list:
        count = count + 1
        # split the record by the commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        cummulative_average = (cummulative_average + n.train(inputs, targets, count, cummulative_average)) / count
    print("")
    pass

print("")
print("# load the mnist test data CSV file into a list")
test_data_file = open("../data/mnist/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

#
print("")
print("# test the neural network")
print("# scorecard for how well the network performs")
scorecard = []

print("")
print("# go through all the records in the test data set")
for record in test_data_list:
    # split the record by the commas
    all_values = record.split(',')
    # correct answer is the first value
    correct_label = int(all_values[0])
    #print (correct_label, " :correct label")
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    #print(label, " :network's answer")
    # append correct or incorrect to list
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

print("")

print("")
print("# calculate the performance score, the fraction of correct answers")
scorecard_array = numpy.asarray(scorecard)
print("hidden_nodes: ", hidden_nodes)
print("learning_rate: ", learning_rate)
print("performance = ", scorecard_array.sum() / scorecard_array.size)

print("")
print("Test with miscellaneous created images")

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('data/five-1.png'):

    # use the filename to set the correct label
    label = 1

    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.v3.imread(image_file_name, mode='F')

    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))

    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)

    pass

# test the neural network with our own images
# record to test
item = 0

# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass

print("hidden_nodes: ", hidden_nodes)
print("learning_rate: ", learning_rate)







