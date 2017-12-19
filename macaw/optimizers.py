import math
import numpy as np

__all__ = ['GradientDescent', 'CoordinateDescent', 'MajorizationMinimization']


class Optimizer(object):
    """A simple base class for an optimizer."""

    def save_state(self, x0, funval, n_iter, status):
        self.x = x0
        self.funval = funval
        self.niters = n_iter
        self.status = status


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

    def __init__(self, fun, gradient, gamma=1e-3):
        self.fun = fun
        self.gradient = gradient
        self.gamma = gamma

    def compute(self, x0, fun_args=(), n=1000, xtol=1e-6, ftol=1e-9, gtol=1e-6):
        fun = _wrap_function(self.fun, fun_args)
        fun_prime = _wrap_function(self.gradient, fun_args)

        i = 0
        while i < n:
            x_tmp = x0
            fun_before = fun(x0)
            grad = fun_prime(x0)

            if math.sqrt(np.sum(grad * grad)) < gtol:
                msg = ("Success: norm of the gradient is less than {}"
                       .format(gtol))
                self.save_state(x0, fun_before, i+1, msg)
                break

            x0 = x0 - self.gamma * grad
            fun_after = fun(x0)
            grad_diff = fun_prime(x0) - grad
            self.gamma = np.dot(x0 - x_tmp, grad_diff) / np.dot(grad_diff, grad_diff)

            if abs((fun_after - fun_before) / (1.+fun_before)) < ftol:
                msg = ("Success: loss function has not changed by {} since"
                       " the previous iteration".format(ftol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            if (abs((x_tmp - x0) / (1.+x0)) < xtol).all():
                msg = ("Success: parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, fun_after, i+1, msg)
                break
            i += 1

        if i == n:
            msg = ("Failure: max. number of iterations ({}) reached."
                   " The algorithm may not have converged.".format(n))
            self.save_state(x0, fun_after, n, msg)


class CoordinateDescent(Optimizer):
    """Implements a sequential coordinate descent algorithm to find local
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

    def __init__(self, fun, gradient, gamma=1e-3):
        self.fun = fun
        self.gradient = gradient
        self.gamma = gamma

    def compute(self, x0, fun_args=(), n=1000, xtol=1e-6, ftol=1e-6, gtol=1e-6):
        fun = _wrap_function(self.fun, fun_args)
        fun_prime = _wrap_function(self.gradient, fun_args)

        x0 = np.asarray(x0)
        i, j, d = 0, 0, len(x0)
        while i < n:
            x0_aux = np.copy(x0)
            total_fun_before = fun(x0_aux)
            k = 0
            while j < d:
                x_tmp = np.copy(x0)
                fun_before = fun(x_tmp)
                grad = fun_prime(x_tmp)[j]

                if math.sqrt(np.sum(grad * grad))/d < gtol:
                    j += 1
                    continue

                x0[j] = x0[j] - self.gamma * grad
                fun_after = fun(x0)
                self.gamma = (x0[j] - x_tmp[j]) / (fun_prime(x0)[j] - grad)

                if (abs((fun_after - fun_before) / (1.+fun_before)) < ftol
                    or abs((x_tmp[j] - x0[j]) / (1.+x0[j])) < xtol or k == n):
                    j += 1
                    continue
                k += 1

            total_fun_after = fun(x0)
            if (abs((total_fun_after - total_fun_before) / (1.+total_fun_before)) < ftol):
                msg = ("Success: loss function has not changed by {} since"
                       " the previous iteration".format(ftol))
                self.save_state(x0, total_fun_after, i+1, msg)
                break

            if (abs((x0_aux - x0) / (1.+x0)) < xtol).all():
                msg = ("Success: parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, total_fun_after, i+1, msg)
                break

            total_grad = fun_prime(x0)
            if math.sqrt(np.sum(total_grad * total_grad))/d < gtol:
                msg = ("Sucess: mean norm of the gradient has not changed by"
                       " {} since the previous iteration.".format(gtol))
                self.save_state(x0, total_fun_after, i+1, msg)
                break

            if j == d:
                j = 0
            i += 1

        if i == n:
            msg = ("Failure: max. number of iterations ({}) reached."
                   " The algorithm may not have converged.".format(n))
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

            * 'gd' : Gradient Descent
            * 'cd' : Coordinate Descent
    kwargs : dict
        Keyword arguments to be passed to the optimizer.
    """

    def __init__(self, fun, optimizer='gd', **kwargs):
        self.fun = fun
        opts = {'gd': GradientDescent, 'cd': CoordinateDescent}
        self.optimizer = opts[optimizer](self.fun.surrogate_fun,
                                         self.fun.gradient_surrogate, **kwargs)

    def compute(self, x0, n=1000, xtol=1e-6, ftol=1e-9, **kwargs):
        i = 0
        while i < n:
            x_tmp = x0
            fun_before = self.fun.evaluate(x0)
            self.optimizer.compute(x0=x0, fun_args=x0, **kwargs)
            x0 = self.optimizer.x
            fun_after = self.fun.evaluate(x0)

            if abs((fun_after - fun_before) / (1.+fun_before)) < ftol:
                msg = ("Success: loss function has not changed by {} since"
                       " the previous iteration".format(ftol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            if (abs((x_tmp - x0) / (1.+x0)) < xtol).all():
                msg = ("Success: parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, fun_after, i+1, msg)
                break

            i += 1

        if i == n:
            msg = ("Failure: max. number of iterations ({}) reached."
                   " The algorithm may not have converged.".format(n))
            self.save_state(x0, fun_after, n, msg)


def _wrap_function(function, args):
    def function_wrapper(wrapper_args):
        if len(args) > 0:
            return function(wrapper_args, args)
        else:
            return function(wrapper_args)
    return function_wrapper
