import numpy as np
from oktopus.models import LineModel
from numpy.testing import assert_allclose
from ..objective_functions import L1Norm, L2Norm
from ..optimizers import GradientDescent, MajorizationMinimization


def test_fitting_line():
    # generate fake data
    np.random.seed(0)
    x = np.linspace(0, 10, 200)
    fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    # build the model
    my_line = LineModel(x)
    # build the objective function
    l2norm = L2Norm(fake_data, my_line)
    # perform optimization
    gd = GradientDescent(l2norm.evaluate, l2norm.gradient)
    gd.compute(x0=(1., 1.))
    assert_allclose(gd.x, [3., 10.], rtol=1e-1)

    l1norm = L1Norm(fake_data, my_line)
    mm = MajorizationMinimization(l1norm)
    mm.compute(x0=(1., 1.))
