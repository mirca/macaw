.. macaw documentation master file, created by
   sphinx-quickstart on Fri Nov 17 08:53:26 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================
Welcome to macaw!
=================
|ci-badge| |cov-badge|

.. |ci-badge| image:: https://travis-ci.org/mirca/macaw.svg?branch=master
.. |cov-badge| image:: https://codecov.io/gh/mirca/macaw/branch/master/graph/badge.svg

**macaw** is a colorful long-tailed package for beautiful Majorization-Minimization
applied to Machine Learning

Majorization-minimization is all about inequalities such as

1. *Jensen's Inequality*:

.. math::

    \phi\left(\mathbb{E}\left[X\right]\right) \leq \mathbb{E}\left[\phi\left(X\right)\right],
for convex :math:`\phi`.

2. *Cauchy-Schwarz Inequality*:

.. math::

    \mathbb{E}^2\left[XY\right] \leq \mathbb{E}\left[X^2\right]\mathbb{E}\left[Y^2\right]

3. *Arithmetic and Geometric Means*:

.. math::

    \mathbb{E}\left[\left(X + Y\right)^2\right] \leq 4\mathbb{E}\left[XY\right]

Those inequalities are applied to complicated objective functions in order to find upper bounds (majorize),
which are subsequently minimized. Iterating this procedure has been proven to be powerful tool for
optimization problems that arise often in signal processing and machine learning ;)

See, for instance, `Majorization-Minimization Algorithms in Signal Processing, Communications, and Machine Learning <http://ieeexplore.ieee.org/document/7547360/>`_ by Y. Sun, P. Babu, and D. P. Palomar.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

*************
Documentation
*************

.. toctree::
    :maxdepth: 1

    install
    api/index
    ipython_notebooks/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
