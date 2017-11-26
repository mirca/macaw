from oktopus.models import Model

class LogisticModel(Model):
    def __init__(self, X):
        self.X = X

    def evaluate(self, w):
        return 1 / (1 - np.exp(-np.dot(X, w)))
