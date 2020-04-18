import numpy as np


class Perceptron(object):

    def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return np.sign(summation)

    def train_p_rule(self, training_inputs, labels):  #PERCEPTRON RULE
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

    def train_d_rule(self, training_inputs, labels):  #DELTA RULE
        for _ in range(self.epochs):
            prediction = self.predict(training_inputs)
            self.weights[1:] += np.sum(self.learning_rate * np.expand_dims(labels - prediction, axis=1) * training_inputs, axis=0)
            self.weights[0] += np.sum(self.learning_rate * (labels - prediction), axis=0)
