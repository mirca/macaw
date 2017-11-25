import numpy as np


__all__ = ['GradientDescent', 'MajorizationMinimization']


class Optimizer(object):
    """A simple base class for an optimizer."""

    def save_state(self, x0, funval, n_iter, message):
        self.x = x0
        self.funval = funval
        self.niters = n_iter
        self.message = message


class GradientDescent(Optimizer):
    """Implements a simple gradient descent algorithm to find local
    minimum of functions.

    Attributes
    ----------
    fun : callable
        The function to be minimized
    gradient : callable
        The gradient of the fun
    gamma : float
        Learning rate
    """

    def __init__(self, fun, gradient, gamma=1e-6):
        self.fun = fun
        self.gradient = gradient
        self.gamma = gamma

    def compute(self, x0, fun_args=(), n=100, xtol=1e-6, ftol=1e-6):
        fun = _wrap_function(self.fun, fun_args)
        fun_prime = _wrap_function(self.gradient, fun_args)

        i = 0
        while i < n:
            x_tmp = x0
            fun_before = fun(x0)
            grad = fun_prime(x0)
            x0 = x0 - self.gamma * grad
            fun_after = fun(x0)

            if abs((fun_after - fun_before) / fun_before) < ftol:
                msg = ("Loss function has not changed by {} since the previous"
                       " iteration".format(ftol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            if (abs((x_tmp - x0) / x0) < xtol).all():
                msg = ("Optimal parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            grad_diff = fun_prime(x0) - grad
            self.gamma = np.dot(x0 - x_tmp, grad_diff) / np.dot(grad_diff, grad_diff)
            i += 1

        if i == n:
            msg = ("Max. number of iterations ({}) reached."
                   " The algorithm might not have converged.".format(n))
            self.save_state(x0, fun_after, n, msg)


class MajorizationMinimization(Optimizer):
    """
    Implements a Majorization-Minimization scheme.

    Attributes
    ----------
    fun : object
        Objective function to be minimized. Note that this must be an object
        that contains the following methods::

            * `evaluate`: returns the value of the objective function
            * `surrogate_fun`: returns the value of an appropriate surrogate function
            * `gradient_surrogate`: return the value of the gradient of the surrogate function
    optimizer : str
        Specifies the optimizer to use during the Minimization step. Options are::

            * 'sgd' : Stochastic Gradient Descent
    kwargs : dict
        Keyword arguments to be passed to the optimizer.
    """

    def __init__(self, fun, optimizer='sgd', **kwargs):
        self.fun = fun
        if optimizer == 'sgd':
            self.optimizer = GradientDescent(fun.surrogate_fun,
                                             fun.gradient_surrogate, **kwargs)
        else:
            raise ValueError('optimizer is not recognized, got {}'.format(optimizer))

    def compute(self, x0, n=100, xtol=1e-6, ftol=1e-6, **kwargs):
        i = 0
        while i < n:
            x_tmp = x0
            fun_before = self.fun.evaluate(x0)
            self.optimizer.compute(x0=x0, fun_args=x0, **kwargs)
            x0 = self.optimizer.x
            fun_after = self.fun.evaluate(x0)

            if abs((fun_after - fun_before) / fun_before) < ftol:
                msg = ("Loss function has not changed by {} since the previous"
                       " iteration".format(ftol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            if (abs((x_tmp - x0) / x0) < xtol).all():
                msg = ("Optimal parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            i += 1

        if i == n:
            msg = ("Max. number of iterations ({}) reached."
                   " The algorithm might not have converged.".format(n))
            self.save_state(x0, fun_after, n, msg)


def _wrap_function(function, args):
    def function_wrapper(wrapper_args):
        if len(args) > 0:
            return function(wrapper_args, args)
        else:
            return function(wrapper_args)
    return function_wrapper
