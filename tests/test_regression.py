# test f1 compactibility with reference implementation by scikit learn

import pytest
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score as sklearn_f1_score
from sklearn.metrics import accuracy_score as sklearn_accuracy_score

from skp4.metrics import f1_score, p4_score


''' Series of randomized tests: calculating f1 score and comparing with the result
achieved from the original sklearn implementation.

The last one compares samples average of p4 score with accuracy score (sklearn) for multilabel classification.

Note: since the algorithm is strictly the same in spite of different implementation, we are comparing
floats using "==" instead of old and reasonable numy.isclose()
'''

@pytest.mark.parametrize("rnd_seed", range(32))
def test_f1_regression_binary(rnd_seed):
    rng = np.random.default_rng(seed = (rnd_seed + 11) * 129)
    y_true = rng.integers(low=0, high=2, size=512)
    y_pred = rng.integers(low=0, high=2, size=512)

    assert f1_score(y_true, y_pred) == sklearn_f1_score(y_true, y_pred)


@pytest.mark.parametrize("rnd_seed", range(32))
def test_f1_regression_multiclass(rnd_seed):
    rng = np.random.default_rng(seed = (rnd_seed + 11) * 129)
    y_true = rng.integers(low=0, high=16, size=512)
    y_pred = rng.integers(low=0, high=16, size=512)

    sklearn_micro = sklearn_f1_score(y_true, y_pred, average='micro')
    sklearn_macro = sklearn_f1_score(y_true, y_pred, average='macro')
    sklearn_weighted = sklearn_f1_score(y_true, y_pred, average='weighted')
    
    scores = f1_score(y_true, y_pred)

    assert scores.micro_average == sklearn_micro
    assert scores.macro_average == sklearn_macro
    assert scores.weighted_average == sklearn_weighted

@pytest.mark.parametrize("rnd_seed", range(32))
def test_f1_regression_multilabel(rnd_seed):
    rng = np.random.default_rng(seed = (rnd_seed + 11) * 129)
    y_true = [rng.choice([1,2,3,4,5,6,7], rng.integers(low=3, high=7, size=1)[0]) for _ in range(512)]
    y_pred = [rng.choice([1,2,3,4,5,6,7], rng.integers(low=3, high=7, size=1)[0]) for _ in range(512)]

    mlb = MultiLabelBinarizer()
    y_true_binarized = mlb.fit_transform(y_true)
    y_pred_binarized = mlb.transform(y_pred)

    sklearn_micro = sklearn_f1_score(y_true_binarized, y_pred_binarized, average='micro')
    sklearn_macro = sklearn_f1_score(y_true_binarized, y_pred_binarized, average='macro')
    sklearn_weighted = sklearn_f1_score(y_true_binarized, y_pred_binarized, average='weighted')
    sklearn_samples = sklearn_f1_score(y_true_binarized, y_pred_binarized, average='samples')

    scores = f1_score(y_true, y_pred)

    assert scores.micro_average == sklearn_micro
    assert scores.macro_average == sklearn_macro
    assert scores.weighted_average == sklearn_weighted
    assert scores.samples_average == sklearn_samples


# samples average for multilabels, gives exact accuracy score
@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_multilabel_samples_average_accuracy(rnd_seed):
    rng = np.random.default_rng(seed = (rnd_seed + 11) * 129)

    y_true = rng.choice(range(20), 512)
    y_pred = rng.choice(range(20), 512)

    assert p4_score(y_true, y_pred).samples_average == sklearn_accuracy_score(y_true, y_pred)


