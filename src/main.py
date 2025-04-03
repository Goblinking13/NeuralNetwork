import numpy

from GUI import appGUI
from NeuralNetwork import NeuralNetwork

input_nodes = 28*28
hidden_nodes = 200
outputs_nodes = 10
learning_rating = 0.1


n = NeuralNetwork(input_nodes, hidden_nodes, outputs_nodes, learning_rating)
n.who = numpy.load("../values/w_hidden_output.npy")
n.wih = numpy.load("../values/w_input_hidden.npy")

gui = appGUI(n)


