from abc import abstractmethod
import numpy as np
from macaw.models import LinearModel, LogisticModel, QuadraticModel
from .optimizers import GradientDescent, CoordinateDescent, MajorizationMinimization


__all__ = ['L1Norm', 'L2Norm', 'BernoulliLikelihood', 'Lasso',
           'RidgeRegression', 'LogisticRegression', 'L1LogisticRegression']


class ObjectiveFunction(object):
    """An abstract class for a generic objective function."""

    def __call__(self, theta):
        """Calls :func:`evaluate`"""
        return self.evaluate(theta)

    @abstractmethod
    def evaluate(self, theta):
        """
        Returns the objective function evaluated at theta

        Parameters
        ----------
        theta : ndarray
            parameter vector of the model

        Returns
        -------
        objective_fun : scalar
            Returns the scalar value of the objective function evaluated at
            **params**
        """
        pass

    def fit(self, x0, optimizer='gd', n=1000, xtol=1e-6, ftol=1e-9):
        opts = {'gd': GradientDescent, 'cd': CoordinateDescent}
        optimizer = opts[optimizer]
        self.opt = optimizer(self.evaluate, self.gradient)
        self.opt.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return self.opt


class L1Norm(ObjectiveFunction):
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

    Examples
    --------
    >>> from macaw.objective_functions import L1Norm
    >>> from macaw.optimizers import MajorizationMinimization
    >>> from macaw.models import LinearModel
    >>> import numpy as np
    >>> # generate fake data
    >>> np.random.seed(0)
    >>> x = np.linspace(0, 10, 200)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> # build the model
    >>> my_line = LinearModel(x)
    >>> # build the objective function
    >>> l1norm = L1Norm(fake_data, my_line)
    >>> # perform optimization
    >>> mm = MajorizationMinimization(l1norm)
    >>> mm.compute(x0=(1., 1.))
    >>> # get best fit parameters
    >>> print(mm.x)
    [  2.96016173  10.30580954]
    """

    def __init__(self, y, model):
        self.y = y
        self.model = model

    def __repr__(self):
        return "<L1Norm(model={})>".format(self.model)

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

    def fit(self, x0, n=1000, xtol=1e-6, ftol=1e-9, **kwargs):
        mm = MajorizationMinimization(self, **kwargs)
        mm.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return mm


class L2Norm(ObjectiveFunction):
    r"""Defines the squared L2 norm loss function. L2 norm
    tends to fit the model to the mean trend of the data.

    .. math::

        \arg \min_{w \in \mathcal{W}}  \frac{1}{2}||y - f(X, \mathbf{w})||^{2}_{2}

    Attributes
    ----------
    y : array-like
        Observed data
    model : callable
        A functional form that defines the model
    yerr : scalar or array-like
        Weights or uncertainties on each observed data point

    Examples
    --------
    >>> import numpy as np
    >>> from macaw.objective_functions import L2Norm
    >>> from macaw.optimizers import GradientDescent
    >>> from macaw.models import LinearModel
    >>> # generate fake data
    >>> np.random.seed(0)
    >>> x = np.linspace(0, 10, 200)
    >>> fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    >>> # build the model
    >>> my_line = LinearModel(x)
    >>> # build the objective function
    >>> l2norm = L2Norm(fake_data, my_line)
    >>> # perform optimization
    >>> gd = GradientDescent(l2norm.evaluate, l2norm.gradient)
    >>> gd.compute(x0=(1., 1.))
    >>> # get the best fit parameters
    >>> print(gd.x)
    [  2.96263148  10.32861519]
    """

    def __init__(self, y, model, yerr=1):
        self.y = y
        self.model = model
        self.yerr = yerr

    def __repr__(self):
        return "<L2Norm(model={})>".format(self.model)

    def evaluate(self, theta):
        residual = self.y - self.model(*theta)
        return .5 * np.nansum(residual * residual / (self.yerr * self.yerr))

    def gradient(self, theta):
        grad = self.model.gradient(*theta)
        return - np.nansum((self.y - self.model(*theta)) * grad
                           / (self.yerr * self.yerr), axis=-1)


class RidgeRegression(ObjectiveFunction):
    r"""
    Implements Ridge regression objective function.

    Ridge regression is a specific case of regression in which the
    model is linear, the objective function is the L2 norm,
    and the regularization term is the L2 norm.

    .. math::

        \arg \min_{w \in \mathcal{W}}  \frac{1}{2}||y - X\mathbf{w}||^{2}_{2} + \alpha||\mathbf{w}||^{2}_{2}
    """
    def __init__(self, y, X, alpha=1):
        self.y = y
        self.model = LinearModel(X)
        self.alpha = alpha
        self._l2norm = L2Norm(y=self.y, model=self.model)

    def evaluate(self, theta):
        theta = np.asarray(theta)
        return (2 * self._l2norm(theta) + self.alpha * np.nansum(theta * theta))

    def gradient(self, theta):
        return 2 * (self._l2norm.gradient(theta) + self.alpha * theta)


class Lasso(ObjectiveFunction):
    r"""
    Implements the Lasso objective function.

    Lasso is usually used to estimate sparse coefficients.

    .. math::

        \arg \min_{w \in \mathcal{W}}  \frac{1}{2\cdot n_{\text{samples}}}||y - X\mathbf{w}||^{2}_{2} + \alpha||\mathbf{w}||^{1}_{1}
    """

    def __init__(self, y, X, alpha=1):
        self.y = y
        self.model = LinearModel(X)
        self.alpha = alpha
        self._l2norm = L2Norm(y=self.y, model=self.model)

    def evaluate(self, theta):
        return (self._l2norm(theta) / len(self.y)
                + self.alpha * np.nansum(np.abs(theta)))

    def surrogate_fun(self, theta, theta_n):
        theta = np.asarray(theta)
        abs_n = np.abs(theta_n)
        return (self._l2norm(theta) / len(self.y)
                + .5 * self.alpha * np.nansum(theta * theta / abs_n + abs_n))

    def gradient_surrogate(self, theta, theta_n):
        theta = np.asarray(theta)
        return (self._l2norm.gradient(theta)
                + self.alpha * theta / np.abs(theta_n))

    def fit(self, x0, n=1000, xtol=1e-6, ftol=1e-9, **kwargs):
        mm = MajorizationMinimization(self, **kwargs)
        mm.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return mm


class BernoulliLikelihood(ObjectiveFunction):
    r"""Implements the negative log likelihood function for independent
    (possibly non-identical distributed) Bernoulli random variables.
    This class also contains a method to compute maximum likelihood estimators
    for the probability of a success.

    More precisely, the MLE is computed as

    .. math::
        \arg \min_{\theta \in \Theta} - \sum_{i=1}^{n} y_i\log\pi_i(\mathbf{\theta}) + (1 - y_i)\log(1 - \pi_i(\mathbf{\theta}))

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
    >>> from macaw.models import ConstantModel as constant
    >>> # generate integer fake data in the set {0, 1}
    >>> np.random.seed(0)
    >>> y = np.random.choice([0, 1], size=100)
    >>> # create a model
    >>> p = constant()
    >>> # perform optimization
    >>> ber = BernoulliLikelihood(y=y, model=p)
    >>> result = ber.fit(x0=[0.3])
    >>> # get best fit parameters
    >>> print(result.x)
    [ 0.55999999]
    >>> print(np.mean(y>0)) # theorectical MLE
    0.56
    >>> # get uncertainties on the best fit parameters
    >>> print(ber.uncertainties(result.x))
    [ 0.04963869]
    >>> # theorectical uncertainty
    >>> print(np.sqrt(.56 * .44 / 100))
    0.049638694584
    """

    def __init__(self, y, model):
        self.y = np.asarray(y)
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

    def fisher_information_matrix(self, theta):
        n_params = len(theta)
        fisher = np.empty(shape=(n_params, n_params))
        grad_model = self.model.gradient(*theta)
        model = self.model(*theta)

        for i in range(n_params):
            for j in range(i, n_params):
                fisher[i, j] = (grad_model[i] * grad_model[j] / model).sum()
                fisher[j, i] = fisher[i, j]
        return len(self.y) * fisher / (1 - self.model(*theta))

    def uncertainties(self, theta):
        inv_fisher = np.linalg.inv(self.fisher_information_matrix(theta))
        return np.sqrt(np.diag(inv_fisher))


class LogisticRegression(BernoulliLikelihood):
    r"""Implements a Logistic regression objective function for
    Binary classification.
    """
    def __init__(self, y, X):
        self.X = X
        super().__init__(y, LogisticModel(self.X))
        self._linear_model = LinearModel(self.X)

    def evaluate(self, theta):
        l = self._linear_model(*theta)
        return np.nansum((1 - self.y) * l + np.log1p(np.exp(-l)))

    def gradient(self, theta):
        l_grad = self._linear_model.gradient(*theta)
        f, f_grad = self.model(*theta), self.model.gradient(*theta)
        return np.nansum((1 - self.y) * l_grad - f_grad / f, axis=-1)

    def surrogate_fun(self, theta, theta_n):
        n = len(self.y)
        f, fn = self.model(*theta), self.model(*theta_n)
        l, ln = self._linear_model(*theta), self._linear_model(*theta_n)

        return np.nansum((1 - self.y) * l + np.log1p(np.exp(-ln))) - n * np.log(n / np.nansum(fn / f))

    def gradient_surrogate(self, theta, theta_n):
        n = len(self.y)
        l = self._linear_model(*theta)
        f, fn = self.model(*theta), self.model(*theta_n)
        l_grad = self._linear_model.gradient(*theta)

        return np.nansum(((1 - self.y) - n * fn * np.exp(-l) / np.nansum(fn / f)) * l_grad,
                         axis=-1)

    def predict(self, X):
        model = LogisticModel(X)
        return np.round(model(*self.opt.x))


class L1LogisticRegression(ObjectiveFunction):
    r"""Implements a Logistic regression objective function with
    L1-norm regularization for Binary classification.
    """

    def __init__(self, y, X, alpha=.1):
        self.y = y
        self.X = X
        self._logistic = LogisticRegression(y, X)
        self.alpha = alpha

    def evaluate(self, theta):
        return self._logistic.evaluate(theta) + self.alpha * np.nansum(np.abs(theta))

    def surrogate_fun(self, theta, theta_n):
        theta = np.asarray(theta)
        abs_n = np.abs(theta_n)
        return (self._logistic.surrogate_fun(theta, theta_n)
                + .5 * self.alpha * np.nansum(theta * theta / abs_n + abs_n))

    def gradient_surrogate(self, theta, theta_n):
        theta = np.asarray(theta)
        return (self._logistic.gradient_surrogate(theta, theta_n)
                + self.alpha * theta / np.abs(theta_n))

    def fit(self, x0, n=10, xtol=1e-6, ftol=1e-9, **kwargs):
        self.mm = MajorizationMinimization(self, **kwargs)
        self.mm.compute(x0=x0, n=n, xtol=xtol, ftol=ftol)
        return self.mm

    def predict(self, X):
        model = LogisticModel(X)
        return np.round(model(*self.mm.x))
