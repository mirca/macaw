import numpy as np
from inspect import signature
from oktopus.loss import LossFunction
from .optimizers import GradientDescent


__all__ = ['L2Norm']


class L2Norm(LossFunction):
    r"""Defines the squared L2 norm loss function. L2 norm
    tends to fit the model to the mean trend of the data.

    .. math::

        {\arg\,min}_{\mathbf{w} \in \mathcal{W}}  \frac{1}{2}||y - f(X, \mathbf{w})||^{2}_{2}

    Attributes
    ----------
    data : array-like
        Observed data
    model : callable
        A functional form that defines the model
    """

    def __init__(self, y, model):
        self.y = y
        self.model = model

    def __repr__(self):
        return "<L2Norm(y={}, model={})>".format(self.y, self.model)

    def evaluate(self, params):
        residual = self.y - self.model(*params)
        return .5 * np.nansum(residual * residual)

    def gradient(self, params):
        _grad = lambda model, argnum, params: model.gradient(*params)[argnum]
        n_params = len(params)
        grad_norm = np.array([])
        for i in range(n_params):
            grad = _grad(self.model, i, params)
            grad_norm = np.append(grad_norm,
                                  - np.nansum((self.y - self.model.evaluate(*params)) * grad)
                                 )
        return grad_norm

    def fit(self, x0=None, n=100, xtol=1e-6, ftol=1e-6):
        if x0 is None:
            n_params = len(signature(self.model.evaluate).parameters)
            x0 = np.random.uniform(low=-1, high=1, size=n_params)
        gd = GradientDescent(self)
        gd.compute(x0, n, xtol, ftol)
        return gd
