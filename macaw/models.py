from abc import abstractmethod
import numpy as np


class Model(object):
    def __call__(self, *params):
        return self.evaluate(*params)

    @abstractmethod
    def evaluate(self, *params):
        pass


class ConstantModel(Model):
    def evaluate(self, c):
        return np.array([c])

    def gradient(self, c):
        return [1.]


class LinearModel(Model):
    def __init__(self, X):
        self.X = np.asarray(X)

    def evaluate(self, *theta):
        if len(self.X.shape) > 1:
            w, b = theta[:-1], theta[-1]
            return np.dot(self.X, w) + b
        else:
            w, b = theta
            return self.X * w + b

    def gradient(self, *theta):
        if len(self.X.shape) > 1:
            X_ = [self.X[:, i] for i in range(self.X.shape[-1])]
            return X_ + [np.ones(self.X.shape[0])]
        else:
            return [self.X, np.ones(len(self.X))]


class LogisticModel(Model):
    def __init__(self, X):
        self.X = X

    def evaluate(self, w):
        return 1 / (1 - np.exp(-np.dot(self.X, w)))

    def gradient(self, w):
        fun = self.evaluate(w)
        return (self.X * np.exp(-np.dot(self.X, w))) * fun ** 2
