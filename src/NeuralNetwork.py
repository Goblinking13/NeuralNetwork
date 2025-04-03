import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        finish_inputs = numpy.dot(self.who, hidden_outputs)
        finish_outputs = self.activation_function(finish_inputs)

        output_errors = targets - finish_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot(output_errors*finish_outputs*(1.0 - finish_outputs), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors*hidden_outputs*(1.0 - hidden_outputs), numpy.transpose(inputs))

        pass

    def query(self, input_list):
        input = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)

        finish_inputs = numpy.dot(self.who, hidden_outputs)
        finish_outputs = self.activation_function(finish_inputs)
        return finish_outputs
        pass


    def backQuery(self,target):

        target_list = numpy.full(10, 0.01)
        target_list[target] = 0.99

        finish_outputs = numpy.array(target_list, ndmin=2).T

        finish_inputs = scipy.special.logit(finish_outputs)

        hidden_outputs = numpy.dot(self.who.T, finish_inputs)

        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = scipy.special.logit(hidden_outputs)

        inputs = numpy.dot( self.wih.T, hidden_inputs)
        inputs-= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01


        return inputs
        pass
    pass

