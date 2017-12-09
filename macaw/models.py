from abc import abstractmethod
import numpy as np


__all__ = ['ConstantModel', 'LinearModel', 'LogisticModel', 'QuadraticModel']


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
        return np.array([1.])


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
            return np.array(X_ + [np.ones(self.X.shape[0])])
        else:
            return np.array([self.X, np.ones(len(self.X))])


class QuadraticModel(Model):
    def __init__(self, X):
        self.X = np.asarray(X)

    def evaluate(self, *theta):
        if len(self.X.shape) > 1:
            w, b = theta[:-1], theta[-1]
            return (np.dot(self.X, w) + b) ** 2
        else:
            w, b = theta
            return (self.X * w + b) ** 2

    def gradient(self, *theta):
        if len(self.X.shape) > 1:
            w, b = theta[:-1], theta[-1]
            X_ = [self.X[:, i] for i in range(self.X.shape[-1])]
            return 2 * (np.dot(self.X, w) + b) * np.array(X_ + [np.ones(self.X.shape[0])])
        else:
            w, b = theta
            return 2 * (self.X * w + b) * np.array([self.X, np.ones(len(self.X))])


class LogisticModel(Model):
    def __init__(self, X):
        self.linear = LinearModel(X)

    def evaluate(self, *theta):
        return 1 / (1 + np.exp(-self.linear(*theta)))

    def gradient(self, *theta):
        fun = np.exp(-self.linear(*theta))
        return fun * self.linear.gradient(*theta) / (1 + fun) ** 2
