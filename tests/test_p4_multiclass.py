

import numpy as np
import pytest

from skp4.generators.multilabel_generators import generate_set_from_confusion_matrix
from skp4.metrics import f1_score, p4_score

def test_trivial_p4_multiclass():

    y_true = [1, 2, 3,   1, 2, 3,   1, 2, 3,   1, 2, 3]
    y_pred = [1, 2, 3,   1, 2, 3,   2, 3, 1,   3, 1, 1]
    
    # macro
    # 1: tp = 2, tn = 5, fp = 3, fn = 2  # p4 = 0.5333333333333333
    # 2: tp = 2, tn = 7, fp = 1, fn = 2  # p4 = 0.6746987951807228
    # 3: tp = 2, tn = 6, fp = 2, fn = 2  # p4 = 0.6

    avg = np.mean([0.6746987951807228, 0.5333333333333333, 0.6])
    
    assert np.isclose(p4_score(y_true, y_pred).macro_average, avg)
    
    # all the weights are the same: weighted == macro
    assert np.isclose(p4_score(y_true, y_pred).weighted_average, avg)

    # micro
    # tp = 6, tn = 18, fp = 6, fn = 6
    # p4 micro = 0.6
    assert np.isclose(p4_score(y_true, y_pred).micro_average, 0.6)

    # accuracy vel samples avg
    # ok = 6, nok = 6
    assert np.isclose(p4_score(y_true, y_pred).samples_average, 0.5)

@pytest.mark.parametrize("rnd_seed", range(32))
def test_perfect_match_p4_multiclass(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(low=0, high=10, size=512)
    result = p4_score(y_true, y_true)
    assert result.micro_average == 1
    assert result.macro_average == 1
    assert result.weighted_average == 1
    assert result.samples_average == 1

@pytest.mark.parametrize("rnd_seed", range(32))
def test_zero_match_p4_multiclass(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(low=0, high=10, size=512)
    y_pred = y_true + 1
    result = p4_score(y_true, y_pred)
    assert result.micro_average == 0
    assert result.macro_average == 0
    assert result.weighted_average == 0
    assert result.samples_average == 0


@pytest.mark.parametrize("rnd_seed", range(32))
def test_close_zero_match_p4_multiclass(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(low=0, high=10, size=10000)
    
    # everything contains false prediction except the first sample
    y_pred = y_true + 1
    y_pred[0] = y_true[0]

    result = p4_score(y_true, y_pred)
    assert np.isclose(result.micro_average, 0, atol=0.001)
    assert np.isclose(result.macro_average, 0, atol=0.001)
    assert np.isclose(result.weighted_average, 0, atol=0.001)
    assert np.isclose(result.samples_average, 0, atol=0.001)


@pytest.mark.parametrize("rnd_seed", range(32))
def test_label_order_invariance(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    
    y_true = rng.integers(low=0, high=10, size=512)
    y_pred = rng.integers(low=0, high=10, size=512)

    first_result = p4_score(y_true, y_pred)
    
    # reorder both: y_true, y_pred the same way
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_true)

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_pred)

    second_result = p4_score(y_true, y_pred)

    # expect the same result
    assert first_result == second_result
    
def test_missing_class():
    rng = np.random.default_rng(seed=31415)
    y_true = rng.integers(low=0, high=10, size=10000)
    y_pred = y_true.copy()
    
    # divert the last element to the class that does not exist in y_pred
    y_true[-1] = 100
    
    # shouldn't throw exception
    result = p4_score(y_true, y_pred)

    assert np.isclose(result.micro_average, 1, atol=0.1)
    assert np.isclose(result.macro_average, 1, atol=0.1)
    assert np.isclose(result.weighted_average, 1, atol=0.1)
    assert np.isclose(result.samples_average, 1, atol=0.1)

    assert result.micro_average != 1
    assert result.macro_average != 1
    assert result.weighted_average != 1
    assert result.samples_average != 1


def test_gradual_deterioration():
    rng = np.random.default_rng(seed=31415)

    dsize = 10000

    y_true = rng.integers(low=0, high=10, size=dsize)
    y_pred = y_true.copy()

    metrics = p4_score(y_true, y_pred)

    # we should observe gradual deterioration of all the metrics
    # as we are adding random noise to y_pred
    for _ in range(10):
        indices = rng.integers(low=0, high=dsize, size=100)
        # divert classes on random positions
        y_pred[indices] = (y_pred[indices] + 1) % 10

        new_metrics = p4_score(y_true, y_pred)
        
        assert new_metrics.micro_average < metrics.micro_average
        assert new_metrics.macro_average < metrics.macro_average
        assert new_metrics.weighted_average < metrics.weighted_average
        assert new_metrics.samples_average < metrics.samples_average

        metrics = new_metrics

@pytest.mark.parametrize("rnd_seed", range(32))
def test_macro_vs_weighted(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    dsize = 10000
    dominating_class = 1
    dominating_size_size = 9000
    
    y_true = rng.integers(low=0, high=10, size=dsize)
    y_true[np.arange(dominating_size_size)] = dominating_class

    y_pred = y_true.copy()
    y_pred[np.arange(dominating_size_size)] = dominating_class + 1
    metrics = p4_score(y_true, y_pred)
    
    # macro average should be bigger than weighted since the prediction
    # for dominating class is wrong
    assert metrics.macro_average > metrics.weighted_average


@pytest.mark.parametrize("rnd_seed", range(32))
def test_alarming_specificity(rnd_seed):
    cm = np.array([
        [8991,  9,   0],
        [950,  50,   0],
        [0,   0,    50]
    ])
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set_from_confusion_matrix(cm, seed=seed)

    p4_metrics = p4_score(y_true, y_pred)
    f1_metrics = f1_score(y_true, y_pred)
    assert p4_metrics.macro_average < f1_metrics.macro_average
    assert p4_metrics.weighted_average < f1_metrics.weighted_average


# tp=8955 tn=45 fp=5 fn=995 # https://arxiv.org/abs/2210.11997
@pytest.mark.parametrize("rnd_seed", range(32))
def test_alarming_npv(rnd_seed):
    cm = np.array([
        [8955,  995,    0],
        [5,      45,    0],
        [0,       0,   50]
    ])
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set_from_confusion_matrix(cm, seed=seed)
    p4_metrics = p4_score(y_true, y_pred)
    f1_metrics = f1_score(y_true, y_pred)
    assert p4_metrics.macro_average < f1_metrics.macro_average
    assert p4_metrics.weighted_average < f1_metrics.weighted_average
