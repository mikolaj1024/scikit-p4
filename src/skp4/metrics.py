

import numpy as np
from skp4.dataset import Dataset
from skp4.formulas.f1 import F1Mixin
from skp4.formulas.p4 import P4Mixin
from skp4.calculation import ScoreCalculation
from skp4.result import MultiResult



def p4_score(y_true, y_pred, *, zero_division='warn') -> np.number | MultiResult:
    '''Calculates the P4 metric for binary, multiclass, and multilabel classification tasks.
    The classifier type is detected automatically.
    
    Parameters:

    `y_true` - true result

    `y_pred` - predicted result

    `zero_division` - determines behavior on division by zero case (compatible witih scikit-learn package):
    * `0.0` - returns zero in case of division by zero 
    * `1.0` - returns one in case of division by zero
    * `np.nan` - returns np.nan in case of division by zero
    * `warn` - returns zero and emits warning in case of division by zero'''
    dataset = Dataset(y_true, y_pred)
    p4score = ScoreCalculation(P4Mixin, dataset=dataset, zero_division=zero_division)
    return p4score.calculate()


def f1_score(y_true, y_pred, *, zero_division='warn') -> np.number | MultiResult:
    dataset = Dataset(y_true, y_pred)
    f1score = ScoreCalculation(F1Mixin, dataset=dataset, zero_division=zero_division)
    return f1score.calculate()