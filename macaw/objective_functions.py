import sys
import numpy as np
from oktopus.likelihood import Likelihood
from oktopus.loss import LossFunction
from .optimizers import GradientDescent, MajorizationMinimization


__all__ = ['L1Norm', 'L2Norm', 'BernoulliLikelihood']


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
    [  2.96298429  10.27857368]
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
        r = self.y - self.model(*theta)
        abs_r = np.abs(self.y - self.model(*theta_n))
        grad_model = self.model.gradient(*theta)
        return - np.nansum(r * grad_model / abs_r, axis=-1)

    def fit(self, x0, n=100, xtol=1e-6, ftol=1e-6):
        mm = MajorizationMinimization(self)
        mm.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return mm


class L2Norm(LossFunction):
    r"""Defines the squared L2 norm loss function. L2 norm
    tends to fit the model to the mean trend of the data.

    .. math::

        {\arg\,\min}_{\mathbf{w} \in \mathcal{W}}  \frac{1}{2}||y - f(X, \mathbf{w})||^{2}_{2}

    Attributes
    ----------
    y : array-like
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
    [  2.96264043  10.32861654]
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
        grad = self.model.gradient(*theta)
        return - np.nansum((self.y - self.model(*theta)) * grad, axis=-1)

    def fit(self, x0, n=100, xtol=1e-6, ftol=1e-6):
        gd = GradientDescent(self.evaluate, self.gradient)
        gd.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return gd


class BernoulliLikelihood(Likelihood):
    r"""Implements the negative log likelihood function for independent
    (possibly non-identical distributed) Bernoulli random variables.
    This class also contains a method to compute maximum likelihood estimators
    for the probability of a success.

    More precisely, the MLE is computed as

    .. math::
        {\arg\,\min}_{\bm{\theta} \in \Theta} - \sum_{i=1}^{n} y_i\log(\pi(\bm{\theta})) + (1 - y_i)\log(1 - \pi(\bm{\theta}))

    Attributes
    ----------
    y : array-like
        Observed data
    model : callable
        A functional form that defines the model for the probability of success

    Examples
    --------
    >>> import numpy as np
    >>> from macaw import BernoulliLikelihood
    >>> from oktopus.models import ConstantModel as constant
    >>> # generate integer fake data in the set {0, 1}
    >>> np.random.seed(0)
    >>> y = np.random.choice([0, 1], size=100)
    >>> p = constant()
    >>> logL = BernoulliLikelihood(y=y, model=p)
    >>> result = logL.fit(x0=[0.3])
    >>> print(result.x)
    [ 0.55999924]
    """

    def __init__(self, y, model):
        self.y = y
        self.model = model

    def evaluate(self, theta):
        model_theta = self.model(*theta)
        return - np.nansum(self.y * np.log(model_theta)
                           + (1. - self.y) * np.log(1. - model_theta))

    def gradient(self, theta):
        model_theta = self.model(*theta)
        grad = self.model.gradient(*theta)
        return - np.nansum(self.y * grad / model_theta
                           - (1 - self.y) * grad / (1 - model_theta),
                           axis=-1)

    def fit(self, x0, n=100, xtol=1e-6, ftol=1e-6):
        gd = GradientDescent(self.evaluate, self.gradient)
        gd.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return gd

class LinearLogisitcRegression(BernoulliLikelihood):
    def __init__(self, y, X):
        self.y = y
        self.X = X
        self.model = LineModel(X)
