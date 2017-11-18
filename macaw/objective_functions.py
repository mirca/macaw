import numpy as np
from inspect import signature
from oktopus.loss import LossFunction
from .optimizers import GradientDescent, MajorizationMinimization


__all__ = ['L1Norm', 'L2Norm']


class L1Norm(LossFunction):
    r"""Defines the L1 Norm loss function. L1 norm is usually useful
    to optimize the "median" model, i.e., it is more robust to
    outliers than the quadratic loss function.

    .. math::

        \arg \min_{\theta \in \Theta} \sum_k |y_k - f(x_k, \theta)|

    Attributes
    ----------
    y : array-like
        Observed data
    model : callable
        A functional form that defines the model
    regularization : callable
        A functional form that defines the regularization term

    Examples
    --------
    >>> from macaw.objective_functions import L1Norm
    >>> from macaw.optimizers import GradientDescent, MajorizationMinimization
    >>> from oktopus.models import LineModel
    >>> import numpy as np
    >>> # generate fake data
    >>> np.random.seed(0)
    >>> x = np.linspace(0, 10, 200)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> # build the model
    >>> my_line = LineModel(x)
    >>> # build the objective function
    >>> l1norm = L1Norm(fake_data, my_line)
    >>> # perform optimization
    >>> mm = MajorizationMinimization(l1norm)
    >>> mm.compute(x0=(1., 1.))
    >>> print(mm.x)
    [  2.96298344  10.27857897]
    """

    def __init__(self, y, model):
        self.y = y
        self.model = model

    def __repr__(self):
        return "<L1Norm(y={}, model={})>".format(self.y, self.model)

    def evaluate(self, theta):
        return np.nansum(np.absolute(self.y - self.model(*theta)))

    def surrogate_fun(self, theta, theta_n):
        """Evaluates a surrogate function that majorizes the L1Norm."""
        r = self.y - self.model(*theta)
        abs_r = np.abs(self.y - self.model(*theta_n))
        return .5 * np.nansum(r * r / abs_r + abs_r)

    def gradient_surrogate(self, theta, theta_n):
        """Computes the gradient of the surrogate function."""
        _grad = lambda model, argnum, theta: model.gradient(*theta)[argnum]
        n_theta = len(theta)
        r = self.y - self.model(*theta)
        abs_r = np.abs(self.y - self.model(*theta_n))
        grad_norm = np.array([])
        for i in range(n_theta):
            grad = _grad(self.model, i, theta)
            grad_norm = np.append(grad_norm,
                                  - np.nansum(r * grad / abs_r)
                                 )
        return grad_norm

    def fit(self, x0=None, n=100, xtol=1e-6, ftol=1e-6):
        if x0 is None:
            n_theta = len(signature(self.model.evaluate).parameters)
            x0 = np.random.uniform(low=-1, high=1, size=n_theta)
        mm = MajorizationMinimization(self)
        mm.compute(x0, n, xtol, ftol)
        return mm

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

    Examples
    --------
    >>> import numpy as np
    >>> from macaw.objective_functions import L2Norm
    >>> from macaw.optimizers import GradientDescent
    >>> from oktopus.models import LineModel
    >>> # generate fake data
    >>> np.random.seed(0)
    >>> x = np.linspace(0, 10, 200)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> # build the model
    >>> my_line = LineModel(x)
    >>> # build the objective function
    >>> l2norm = L2Norm(fake_data, my_line)
    >>> # perform optimization
    >>> gd = GradientDescent(l2norm.evaluate, l2norm.gradient)
    >>> gd.compute(x0=(1., 1.))
    >>> print(gd.x)
    [  2.96264076  10.32861659]
    """

    def __init__(self, y, model):
        self.y = y
        self.model = model

    def __repr__(self):
        return "<L2Norm(y={}, model={})>".format(self.y, self.model)

    def evaluate(self, theta):
        residual = self.y - self.model(*theta)
        return .5 * np.nansum(residual * residual)

    def gradient(self, theta):
        _grad = lambda model, argnum, theta: model.gradient(*theta)[argnum]
        n_theta = len(theta)
        grad_norm = np.array([])
        for i in range(n_theta):
            grad = _grad(self.model, i, theta)
            grad_norm = np.append(grad_norm,
                                  - np.nansum((self.y - self.model.evaluate(*theta)) * grad)
                                 )
        return grad_norm

    def fit(self, x0=None, n=100, xtol=1e-6, ftol=1e-6):
        if x0 is None:
            n_theta = len(signature(self.model.evaluate).parameters)
            x0 = np.random.uniform(low=-1, high=1, size=n_theta)
        gd = GradientDescent(self.evaluate, self.gradient)
        gd.compute(x0, n, xtol, ftol)
        return gd
