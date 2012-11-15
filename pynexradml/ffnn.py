from __future__ import division
import numpy as np
import json, random

class LogisticActivationFunction(object):
    def __init__(self):
        pass
    def activate(self, x):
        return 1 / (1 + np.exp(0 - x))
    def derivative(self, x):
        return np.multiply(x,  (1 - x))

class HyperbolicTangentActivationFunction(object):
    def __init__(self, a=1.7159, b=0.666666):
        self.a = a
        self.b = b
    def activate(self, x):
        return self.a * np.tanh(self.b * x)
    def derivative(self, x):
        return (self.a * self.b) * (1 - (np.multiply(x, x) / (self.a * self.a)))

class FeedForwardNeuralNet(object):
    def __init__(self, layers=None, sigmoids=None):
        self.weights = []
        self.prevDeltaW = []
        if sigmoids != None:
            self.sigmoids = sigmoids
        elif layers != None:
            self.sigmoids = [HyperbolicTangentActivationFunction() for i in range(1, len(layers))]
        else: 
            self.sigmoids = []
        if layers != None:
            if len(layers) < 2:
                raise ValueError('Network contain at least two layers')
            for i in range(1, len(layers)):
                self.weights.append(np.matrix(np.random.rand(layers[i - 1] + 1, layers[i])) - 0.5)
                self.prevDeltaW.append(np.matrix(np.zeros((layers[i - 1] + 1, layers[i]))))

        """
        Only used for basic examples. It is assumed that practicle applications will
        override get_learning_instance and get_validation_instance
        """
        self.learning_data = []
        self.validation_data = []
        self.activations = []
        self.accum_mse = []
        self.shuffle = False
        self.momentum = 0
        self.verbose = False

    def activate(self, inputs):
        self.activations = [np.matrix(inputs)]
        for i in range(len(self.weights)):
            self.activations.append(self.sigmoids[i].activate(np.hstack([self.activations[-1], [[1]]]) * self.weights[i]))
        return self.activations[-1]

    def backProp(self, target, learning):
        sigmas = []
        for i in reversed(range(len(self.weights))):
            o = self.activations[i+1]
            D = np.matrix(np.diag(np.array(self.sigmoids[i].derivative(o))[0]))
            if len(sigmas) == 0:
                e = o - target
                sigmas.insert(0, D * e)
            else:
                sigmas.insert(0, D * self.weights[i+1][:-1] * sigmas[0])
        for i in range(len(self.weights)):
            deltaW =  np.transpose(-learning * sigmas[i] * np.hstack([self.activations[i], [[1]]]))
            if self.momentum > 0:
                deltaW += self.momentum * self.prevDeltaW[i];
                self.weights[i] = self.weights[i] + deltaW
                self.prevDeltaW[i] = deltaW
            else:
                self.weights[i] = (self.weights[i] + deltaW)

    def learn(self, learning=.03, epochs=100):
        self.accum_mse = []
        for i in xrange(epochs):
            (mse, count) = 0, 0
            for (inputs, target) in self.getLearningData():
                output = self.activate(inputs)
                mse += ((target - output)**2)
                count += 1
                self.backProp(target, learning)
                if self.verbose and count % 10000 == 0:
                    print "%d instances processed for epoch %d" % (count, i)
            mse = mse / count
            self.accum_mse.append(mse)
            print "MSE for epoch %d : %f" % (i, mse)
        print "Final Training SSE = %f" % (self._computeSSE())

    def _computeSSE(self):
        sse = 0.0
        for (inputs, target) in self.getLearningData():
            output = self.activate(inputs)
            sse += ((target - output)**2)
        return sse

    def performSimulatedAnnealing(self, alpha=.1, epochs=100000, temp=1000000, tempStep=2000, tempInterval=250):
        energy = self._computeSSE()
        for i in xrange(epochs):
            print "Energy for iteration %d = %f" % (i, energy)
            if i % tempInterval == 0:
                temp -= tempStep
            for layer in self.weights:
                for index, w in np.ndenumerate(layer):
                    storedWeight = w
                    layer[index] = w + (alpha * (random.random() * 2 - 1))
                    newEnergy = self._computeSSE()
                    if newEnergy > energy:
                        deltaE = newEnergy - energy
                        pE = 1 / (1 + np.exp(deltaE / temp))
                        pAccept = random.random()
                        if pAccept < pE:
                            energy = newEnergy
                        else:
                            layer[index] = storedWeight
                    else:
                        energy = newEnergy

    def serialize(self):
        properties = {
            'weights' : [x.tolist() for x in self.weights]
            }
        return properties

    def deserialize(self, properties):
        self.weights = [np.matrix(x) for x in properties['weights']]

    def save(self, filename):
        data_str = json.dumps(self.serialize())
        with open(filename, 'w') as f:
            f.write(data_str)

    @staticmethod
    def load(filename):
        with open(filename, 'r') as f:
            data_str = f.read()
        properties = json.loads(data_str)
        result = FeedForwardNeuralNet()
        result.deserialize(properties)
        result.sigmoids = [HyperbolicTangentActivationFunction() for i in range(len(result.weights))]
        return result

    def validate(self, callback=None):
        (mse, count) = 0, 0
        for (inputs, target) in self.getValidationData():
            output = self.activate(inputs)
            mse += ((target - output)**2)
            count += 1
            if callback != None:
                callback(inputs, target, output)
        self.mse = mse / count
        print "  - MSE = %f" % self.mse
        return self.mse

    def getLearningData(self):
        if self.shuffle:
            np.random.shuffle(self.learning_data)
        for i in xrange(len(self.learning_data)):
            yield (self.learning_data[i, :-1], self.learning_data[i, -1])

    def getValidationData(self):
        for i in xrange(len(self.validation_data)):
            yield (self.validation_data[i, :-1], self.validation_data[i, -1])

"""
    def printStats(self, tp, tn, fp, fn, mse):
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = 0.0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0.0 if (tp + fn) == 0 else tp / (tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)

        print "---------------------------------"
        print "Classifier Performance:"
        print ""
        print "  Confusion Matrix"
        print "       TP    |    FP     " 
        print "  +---------------------+"
        print "  |%10d|%10d|" % (tp, fp)
        print "  |%10d|%10d|" % (fn, tn)
        print "  +---------------------+"
        print "       FN    |    TN     " 
        print ""
        print "  - Accuracy = %.3f" % accuracy
        print "  - Precision = %.3f" % precision
        print "  - Recall = %.3f" % recall
        print "  - F1 = %.3f" % f1
        print "  - MSE = %.3f" % mse
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
        
        self.printStats(tp, tp, fp, fn, self.mse)

        return (self.mse, tp, tn, fp, fn)
    """
