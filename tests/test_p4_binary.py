import numpy as np
import pytest

from skp4.generators.binary_generators import construct_set, generate_set
from skp4.metrics import p4_score, f1_score

''' Testing binary P4 classiffication metric. Includes 4 "alarming" cases 
as described in [1]. Some tests are repeated automatically with different seed for pseudo-random numbers generator.

[1] https://arxiv.org/abs/2210.11997
'''

def test_trivial_p4_binary_case():
    y_true = [1, 1, 1, 1, 0, 0]
    y_pred = [1, 1, 1, 0, 1, 0]

    # tp=3 tn=1 fp=1 fn=1
    assert np.isclose(p4_score(y_true, y_pred), 0.6)

def test_p4_perfect_match():
    seed = 314151
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(0, 2, 10000)
    assert p4_score(y_true, y_true) == 1.0

def test_p4_close_zero():
    seed = 314151
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(0, 2, 10000)
    y_pred = -1 * y_true + 1
    y_pred[0] = y_true[0]
    assert np.isclose(p4_score(y_true, y_pred), 0.0)


def test_p4_zero():
    seed = 314151
    rng = np.random.default_rng(seed=seed)
    y_true = rng.integers(0, 2, 10000)
    y_pred = -1 * y_true + 1
    assert p4_score(y_true, y_pred) == 0


@pytest.mark.parametrize("rnd_seed", range(32))
def test_label_order_invariance(rnd_seed):
    seed = (rnd_seed + 11) * 129
    rng = np.random.default_rng(seed=seed)
    
    y_true = rng.integers(low=0, high=2, size=512)
    y_pred = rng.integers(low=0, high=2, size=512)

    first_result = p4_score(y_true, y_pred)
    
    # reorder both: y_true, y_pred the same way
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_true)

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_pred)

    second_result = p4_score(y_true, y_pred)

    # expect the same result
    assert first_result == second_result


@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_binary_monte_carlo_cases(rnd_seed):
    rng = np.random.default_rng(seed=rnd_seed)
    tp, tn, fp, fn = rng.integers(low=10, high=10000, size=4)
    y_true, y_pred = construct_set(tp, tn, fp, fn)
    ref = (4 * tp * tn) / (4 * tp * tn + (tp + tn) * (fp + fn))
    assert np.isclose(p4_score(y_true, y_pred), ref)


@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_alarming_precision(rnd_seed):
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set(n=10000, positives_fraction=0.001, tpr=0.95, tnr=0.95, seed=seed)
    assert p4_score(y_true, y_pred) < 0.1
    assert f1_score(y_true, y_pred) < 0.1

@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_alarming_recall(rnd_seed):
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set(n=10000, positives_fraction=0.1, tpr=0.005, tnr=0.9999, seed=seed)
    assert p4_score(y_true, y_pred) < 0.1
    assert f1_score(y_true, y_pred) < 0.1

@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_alarming_specificity(rnd_seed):
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set(n=10000, positives_fraction=0.9, tpr=0.999, tnr=0.005, seed=seed)
    assert p4_score(y_true, y_pred) < 0.1
    assert f1_score(y_true, y_pred) > 0.9

@pytest.mark.parametrize("rnd_seed", range(32))
def test_p4_alarming_npv(rnd_seed):
    seed = (rnd_seed + 11) * 129
    y_true, y_pred = generate_set(n=10000, positives_fraction=0.9999, tpr=0.9, tnr=0.9, seed=seed)
    assert p4_score(y_true, y_pred) < 0.1
    assert f1_score(y_true, y_pred) > 0.9
    
    
