import numpy as np

__all__ = ['GradientDescent']


class GradientDescent(object):
    """Implements a simple gradient descent algorithm to find local
    minimum of functions.

    Attributes
    ----------
    loss_function : callable
        A callable that has a ``gradient`` method such that
        ``loss_function.gradient(*args)`` return the gradient value evaluated
        at ``args``.
    """

    def __init__(self, loss_function, learning_rate=.1):
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def compute(self, x0, n=100, xtol=1e-6, ftol=1e-6):
        i = 0
        while i < n:
            loss_before = self.loss_function(x0)
            x_tmp = x0
            grad = self.loss_function.gradient(x0)
            x0 = x0 - self.learning_rate * grad
            loss_after = self.loss_function(x0)

            if abs((loss_after - loss_before) / loss_before) < ftol:
                msg = ("Loss function has not changed by {} since the previous"
                       " iteration".format(ftol))
                self.save_state(x0, loss_after, i+1, msg)
                break

            if (abs((x_tmp - x0) / x0) < xtol).all():
                msg = ("Optimal parameters have not changed by {} since"
                       " the previous iteration.".format(xtol))
                self.save_state(x0, loss_after, i+1, msg)
                break

            grad_diff = self.loss_function.gradient(x0) - grad
            self.learning_rate = np.dot(x0 - x_tmp, grad_diff) / np.dot(grad_diff, grad_diff)

            i += 1

        if i == n:
            msg = ("Max. number of iterations ({}) reached."
                   " The algorithm might not have converged.".format(n))
            self.save_state(x0, loss_after, n, msg)

    def save_state(self, x0, funval, n_iter, message):
        self.x = x0
        self.fun = funval
        self.niters = n_iter
        self.message = message

