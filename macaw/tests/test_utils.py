import numpy as np
from ..optimizers import _wrap_function

def test_wrap_function():
    def sum_multiply(a, b):
        return np.sum(a) * np.array(b)

    def _sum(a):
        return np.sum(a)

    _sum_mult = _wrap_function(sum_multiply, np.array([5]))
    assert _sum_mult([1, 2]) == _sum_mult([2, 1]) == 15

    my_sum = _wrap_function(_sum, ())
    assert my_sum([1, 2]) == my_sum([2, 1]) == 3
