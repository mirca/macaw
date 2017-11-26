import numpy as np
from oktopus.models import LinearModel
from sklearn import datasets, linear_model
from numpy.testing import assert_allclose
from ..objective_functions import L1Norm, L2Norm
from ..optimizers import GradientDescent, MajorizationMinimization


def test_fitting_line():
    # generate fake data
    np.random.seed(0)
    x = np.linspace(0, 10, 200)
    fake_data = x * 3 + 10 + np.random.normal(scale=2, size=x.shape)
    # build the model
    my_line = LinearModel(x)
    # build the objective function
    l2norm = L2Norm(fake_data, my_line)
    # perform optimization
    gd = GradientDescent(l2norm.evaluate, l2norm.gradient)
    gd.compute(x0=(1., 1.))
    assert_allclose(gd.x, [3., 10.], rtol=1e-1)

    l1norm = L1Norm(fake_data, my_line)
    mm = MajorizationMinimization(l1norm)
    mm.compute(x0=(1., 1.))
    assert_allclose(mm.x, [3., 10.], rtol=1e-1)

def test_ordinary_least_squares_against_sklearn():
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()
    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    # Get training set
    diabetes_X_train = diabetes_X[:-20]
    diabetes_y_train = diabetes.target[:-20]
    # Create and train linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(diabetes_X_train, diabetes_y_train)

    # Using macaw
    model_train = LinearModel(diabetes_X_train.reshape(-1))
    l2norm = L2Norm(y=diabetes_y_train.reshape(-1), model=model_train)
    results = l2norm.fit(x0=[0., 0.])

    assert_allclose(regr.coef_, results.x[0])
    assert_allclose(regr.intercept_, results.x[1])
