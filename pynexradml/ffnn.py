from __future__ import division
import numpy as np
import json

class FeedForwardNeuralNet(object):
    def __init__(self, inputs, hidden, linear_output=True):
        self.i_count = inputs
        self.h_count = hidden
        self.linear_output = linear_output
        #network parameters
        self.weights_h = np.matrix(np.random.rand(inputs, hidden))
        self.weights_y = np.matrix(np.random.rand(hidden))
        self.bias_h = np.matrix(np.random.rand(hidden))
        self.bias_y = np.random.rand()
        #network variables
        self.h = np.matrix(np.zeros(hidden))
        self.x = np.matrix(np.zeros(inputs))
        self.sigma_h = np.matrix(np.zeros(hidden))
        self.y = 0
        self.sigma_y = 0
        """
        Only used for basic examples. It is assumed that practicle applications will
        override get_learning_instance and get_validation_instance
        """
        self.learning_data = []
        self.learning_targets = []
        self.validation_data = []
        self.validation_targets = []
        self.activations = []
        self.accum_mse = []

    def serialize(self):
        properties = {
            'linear_output' : self.linear_output,
            'inputs' : self.i_count,
            'hidden' : self.h_count,
            'weights_h' : self.weights_h.tolist(),
            'weights_y' : self.weights_y.tolist(),
            'bias_h' : self.bias_h.tolist(),
            'bias_y' : self.bias_y}
        return properties

    def deserialize(self, properties):
        self.linear_output = properties['linear_output']
        self.i_count = properties['inputs']
        self.h_count = properties['hidden']
        #load network parameters
        self.weights_h = np.matrix(properties['weights_h'])
        self.weights_y = np.matrix(properties['weights_y'])
        self.bias_h = np.matrix(properties['bias_h'])
        self.bias_y = properties['bias_y']
        #initialize network variables
        self.h = np.matrix(np.zeros(self.weights_h.shape[1]))
        self.x = np.matrix(np.zeros(self.weights_h.shape[0]))
        self.sigma_h = np.matrix(np.zeros(self.weights_h.shape[1]))
        self.y = 0
        self.sigma_y = 0
        self.learning_properties = []
        self.learning_targets = []
        self.validation_properties = []
        self.validation_targets = []
        self.activations = []
        self.accum_mse = []

    def save(self, filename):
        data_str = json.dumps(self.serialize())
        with open(filename, 'w') as f:
            f.write(data_str)

    def load(self, filename):
        with open(filename, 'r') as f:
            data_str = f.read()
        properties = json.loads(data_str)
        self.deserialize(properties)

    def activate(self, inputs):
        #copy inputs
        self.x = np.matrix(inputs)
        #hidden layer activation
        self.h = np.tanh((self.x * self.weights_h) + self.bias_h)
        #output layer activation
        output = (self.h * np.transpose(self.weights_y)) + self.bias_y
        if self.linear_output:
            self.y = np.asscalar(output)
        else:
            self.y = np.asscalar(1 / (1 + np.exp(-output)))
        return self.y

    def back_prop(self, target, learning):
        #calculate output layer error
        if self.linear_output:
            self.sigma_y = (target - self.y)
        else:
            self.sigma_y = self.y * (1 - self.y) * (target - self.y)
        #calculate hidden layer error
        self.sigma_h = np.multiply((1 - np.multiply(self.h, self.h)), self.weights_y) * self.sigma_y
        #update output layer weights
        self.weights_y = self.weights_y + (learning * self.h * self.sigma_y)
        #update output layer bias
        self.bias_y = self.bias_y + (learning * self.sigma_y)
        #update hidden layer weights
        self.weights_h = self.weights_h + (learning * np.transpose(self.x) * self.sigma_h)
        #update hidden layer bias
        self.bias_h = self.bias_h + (learning * self.sigma_h)

    def learn(self, learning=.03, epochs=100):
        self.accum_mse = []
        for i in xrange(epochs):
            (mse, count) = 0, 0
            for (inputs, target) in self.get_learning_data():
                output = self.activate(inputs)
                mse += ((target - output)**2)
                count += 1
                self.back_prop(target, learning)
            mse = mse / count
            self.accum_mse.append(mse)
            print "MSE for epoch %d : %f" % (i, mse)

    def printStats(self, tp, tn, fp, fn):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

        print "---------------------------------"
        print "Classifier Performance:"
        print ""
        print "  Confusion Matrix"
        print "  +---------------------+"
        print "  |%10d|%10d|" % (tp, fp)
        print "  |%10d|%10d|" % (fn, tn)
        print "  +---------------------+"
        print ""
        print "  - Accuracy = %.3f" % accuracy
        print "  - Precision = %.3f" % precision
        print "  - Recall = %.3f" % recall
        print "  - F1 = %.3f" % f1
        print "  - MSE = %.3f" % self.mse
        print "---------------------------------"
    
    def validate(self):
        (mse, count, tp, tn, fp, fn) = 0, 0, 0, 0, 0, 0 
        self.activations = []
        for (inputs, target) in self.get_validation_data():
            output = self.activate(inputs)
            self.activations.append(output)
            mse += ((target - output)**2)
            output = 1 if output > 0.5 else 0
            if output == 1 and target == 1:
                tp += 1
            elif output == 1 and target == 0:
                fp += 1
            elif output == 0 and target == 0:
                tn += 1
            elif output == 0 and target == 1:
                fn += 1
            count += 1
        self.mse = mse / count
        
        self.printStats(tp, tp, fp, fn)

        return self.mse

    def get_learning_data(self):
        for i in xrange(len(self.learning_data)):
            yield (self.learning_data[i], self.learning_targets[i])
    def get_validation_data(self):
        for i in xrange(len(self.validation_data)):
            yield (self.validation_data[i], self.validation_targets[i])

if __name__ == '__main__':
    import math
    import random

    import matplotlib
    from pylab import figure, plot, legend, subplot, grid, xlabel, ylabel, show, title
    

    pop_len = 200
    factor = 1.0 / float(pop_len)
    population = [[i, math.sin(float(i) * factor * 10.0) + \
                    random.gauss(float(i) * factor, .2)]
                        for i in range(pop_len)]
    all_inputs = []
    all_targets = []

    def population_gen(population):
        pop_sort = [item for item in population]
        random.shuffle(pop_sort)
        for item in pop_sort:
            yield item

    #   Build the inputs
    for position, target in population_gen(population):
        pos = float(position)
        all_inputs.append([random.random(), pos * factor])
        all_targets.append(target)

    network = FeedForwardNeuralNet(2,10)
    length = len(all_inputs) // 10 * 8
    network.learning_data = all_inputs[0:length]
    network.learning_targets = all_targets[0:length]
    network.validation_data = all_inputs[length:]
    network.validation_targets = all_targets[length:]
    network.learn(learning=0.1, epochs=125)
    mse = network.validate()
    print network.validation_targets

    test_positions = [item[1] * 1000.0 for item in network.validation_data]
    all_targets1 = network.validation_targets
    allactuals = network.activations

    f = figure(figsize=(12,10), dpi=80)
    #   This is quick and dirty, but it will show the results
    subplot(3, 1, 1)
    plot([i[1] for i in population])
    title("Population")
    grid(True)

    subplot(3, 1, 2)
    plot(test_positions, all_targets1, 'bo', label='targets')
    plot(test_positions, allactuals, 'ro', label='actuals')
    grid(True)
    legend(loc='lower left', numpoints=1)
    title("Test Target Points vs Actual Points")

    subplot(3, 1, 3)
    plot(range(1, len(network.accum_mse) + 1, 1), network.accum_mse)
    xlabel('epochs')
    ylabel('mean squared error')
    grid(True)
    title("Mean Squared Error by Epoch")

    show()
    


        
