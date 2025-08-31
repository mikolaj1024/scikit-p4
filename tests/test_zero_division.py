
# all the zero division stuff

import numpy as np
import pytest
from skp4.binary import BinaryScore
from skp4.dataset import Dataset
from skp4.exceptions import SklearnP4Exception
from .helper_metrics import ZenMetricMixin

class ZDScore(ZenMetricMixin, BinaryScore):
    ...

def test_zero_division_scalar():
    dataset = Dataset([1,2,3], [1,2,3])

    zdsc = ZDScore(dataset=dataset, zero_division='warn')
    assert zdsc.division(1, 0) == 0

    zdsc = ZDScore(dataset=dataset, zero_division=0.0)
    assert zdsc.division(1, 0) == 0.0

    zdsc = ZDScore(dataset=dataset, zero_division=1.0)
    assert zdsc.division(1, 0) == 1.0

    zdsc = ZDScore(dataset=dataset, zero_division=np.nan)
    assert np.isnan(zdsc.division(1, 0))

    with pytest.raises(SklearnP4Exception):
        zdsc = ZDScore(dataset=dataset, zero_division=2.0)
        zdsc.division(1, 0)


def test_zero_division_array():
    dataset = Dataset([1,2,3], [1,2,3])

    numerator = np.array([[1,2,3], [4,5,6]])
    denominator = np.array([[1, 0, 3], [4, 5, 0]])

    zdsc = ZDScore(dataset=dataset, zero_division='warn')
    result = zdsc.division(numerator=numerator, denominator=denominator)

    equal = (result == np.array([[1., 0., 1.], [1., 1., 0.]]))    
    assert np.all(equal)

    zdsc = ZDScore(dataset=dataset, zero_division=0)
    result = zdsc.division(numerator=numerator, denominator=denominator)

    equal = (result == np.array([[1., 0., 1.], [1., 1., 0.]]))    
    assert np.all(equal)

    zdsc = ZDScore(dataset=dataset, zero_division=1.0)
    result = zdsc.division(numerator=numerator, denominator=denominator)

    equal = (result == np.array([[1., 1., 1.], [1., 1., 1.]]))
    assert np.all(equal)


    zdsc = ZDScore(dataset=dataset, zero_division=np.nan)
    result = zdsc.division(numerator=numerator, denominator=denominator)

    nans = np.isnan(result)
    equal = (nans == np.array([[False, True, False], [False, False, True]]))
    assert np.all(equal)
