import tensorflow as tf
import math

def lad(X, y, yerr=None, l1_regularizer=0.12, niters=5, rtol=1e-4, session=None):
    """
    L1 norm optimization (or least absolute deviation) with L1 norm
    regularization using Majorization-Minimization.

    Parameters
    ----------
    X : (n, m)-matrix
        Design matrix
    y : (n, 1) matrix
        Vector of observations
    yerr : (n, 1) matrix
        Vector of standard deviations on the observations
    l1_regularizer : float
        Factor to control the regularization strength
    niters : int
        Number of iterations of the Majorization-Minimization
    rtol : float
        Relative tolerance used as an early stopping criterion
    session : tf.Session object

    Returns
    -------
    x : (m, 1) matrix
        Vector of coefficients that minimizes the least absolute deviations
        with L1 regularization.
    """

    if yerr is not None:
        whitening_factor = yerr/math.sqrt(2.)
    else:
        whitening_factor = 1.

    # convert inputs to tensors
    X_tensor = tf.convert_to_tensor(X / whitening_factor, dtype=tf.float64)
    y_tensor = tf.reshape(tf.convert_to_tensor(y / whitening_factor,
                          dtype=tf.float64), (-1, 1))

    with session or tf.Session() as session:
        # solve the OLS problem and use it as initial values for the MM algorithm
        x = tf.matrix_solve_ls(X_tensor, y_tensor, l2_regularizer=l1_regularizer)
        n = 0
        while n < niters:
            reg_factor = tf.norm(x, ord=1)
            l1_factor = tf.sqrt(tf.abs(y_tensor - tf.matmul(X_tensor, x)))

            X_tensor = X_tensor / l1_factor
            y_tensor = y_tensor / l1_factor

            xo = tf.matrix_solve_ls(X_tensor, y_tensor,
                                    l2_regularizer=l1_regularizer/reg_factor)

            rel_err = tf.norm(x - xo, ord=1) / (1. + reg_factor)
            if session.run(rel_err) < rtol:
                return xo
            x = xo
            n += 1
    return xo
