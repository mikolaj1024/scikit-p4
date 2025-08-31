
import numpy as np
import pytest
from skp4.metrics import f1_score, p4_score
from tests.helper_random import generate_random_labels


def test_trivial_p4_multilabel():
    y_true = [[1,2], [1,2], [3]]
    y_pred = [[1,2], [1], [2,3]]
    result = p4_score(y_true, y_pred)

    # macro
    # label 1: tp=2 tn=1 fp=0 fn=0 # p4=1.0
    # label 2: tp=1 tn=0 fp=1 fn=1 # p4=0.0
    # label 3: tp=1 tn=2 fp=0 fn=0 # p4=1.0
    avg = np.mean([1.0, 0.0, 1.0])
    assert np.isclose(result.macro_average, avg)

    wavg = np.average([1.0, 0.0, 1.0], weights=[2, 2, 1])
    assert np.isclose(result.weighted_average, wavg)

    # micro
    # tp=2+1+1 tn=1+0+2 fp=0+1+0 fn=0+1+0 # p4 = 0.7741935483870968
    assert np.isclose(result.micro_average, 0.7741935483870968)

    # samples
    # sample 1: tp=2 tn=1 fp=0 fn=0 # p4=1.0
    # sample 2: tp=1 tn=1 fp=0 fn=1 # p4=0.6666666666666
    # sample 3: tp=1 tn=1 fp=1 fn=0 # p4=0.6666666666666
    savg = np.mean([1.0, 0.6666666666666, 0.6666666666666])
    assert np.isclose(result.samples_average, savg)


def test_below_one_perfect_p4_multilabel_match():
    'case where in spite of perfect prediction, the score is below 1'
    y_true = [[1,2], [1]] 
    y_pred = [[1,2], [1]]
    result = p4_score(y_true, y_pred)

    # macro
    # label 1: tp=2 tn=0 fp=0 fn=0 # p4=0.0 (through div by zero)
    # label 2: tp=1 tn=1 fp=0 fn=0 # p4=1.0
    assert np.isclose(result.macro_average, 0.5)

    # weighted: 2 * 0 + 1 * 1 / 3
    assert np.isclose(result.weighted_average, 0.333333333333)

    # micro
    # tp=2+1 tn=0+1 fp=0+0 fn=0+0 # p4 = 1.0
    assert np.isclose(result.micro_average, 1.0)

    # samples
    # sample 1: tp=2 tn=0 fp=0 fn=0 # p4=0.0 # div by zero
    # sample 2: tp=1 tn=1 fp=0 fn=0 # p4=1.0
    assert np.isclose(result.samples_average, 0.5)

def test_perfect_zerodivision1_p4_multilabel_match():
    '''case like above - where in spite of perfect prediction, the score is below 1
    but using zero_division=1 - vices 1.0 output'''
    y_true = [[1,2], [1]] 
    y_pred = [[1,2], [1]]
    result = p4_score(y_true, y_pred, zero_division=1.0)

    assert np.isclose(result.macro_average, 1.0)
    assert np.isclose(result.weighted_average, 1.0)
    assert np.isclose(result.micro_average, 1.0)
    assert np.isclose(result.samples_average, 1.0)

@pytest.mark.parametrize("rnd_seed", range(32))
def test_perfect_match_micro_average_p4(rnd_seed):
    'micro-average is always 1.0 for perfect match even with zero_division=0.0'
    seed=(rnd_seed + 11) * 129
    y_true = generate_random_labels(nsize=200, total_labels=100, min_num_labels=0, max_num_labels=3, seed=seed)
    y_pred = y_true
    result = p4_score(y_true, y_pred)
    assert np.isclose(result.micro_average, 1.0)


def test_all_wrong_predictions():
    y_true = [[1,2], [3,4], [5,6]]
    y_pred = [[3,4], [5,6], [7,8]]
    result = p4_score(y_true, y_pred)
    
    # hard zeros here expected
    assert result.macro_average == 0
    assert result.weighted_average == 0
    assert result.micro_average == 0
    assert result.samples_average == 0


@pytest.mark.parametrize("rnd_seed", range(128))
def test_low_labels_per_sample(rnd_seed):
    
    seed = (rnd_seed + 11) * 129

    # we are generating dataset with many labels possible (100) but small number of labels per sample
    y_true = generate_random_labels(nsize=200, total_labels=100, min_num_labels=0, max_num_labels=3, seed=seed)
    y_pred = generate_random_labels(nsize=200, total_labels=100, min_num_labels=0, max_num_labels=3, seed=seed)

    f1 = f1_score(y_true, y_pred)
    p4 = p4_score(y_true, y_pred)

    # that gives a lot of TN causing being f1 <= p4    
    assert f1.macro_average <= p4.macro_average
    assert f1.weighted_average <= p4.weighted_average
    assert f1.micro_average <= p4.micro_average
    assert f1.samples_average <= p4.samples_average

@pytest.mark.parametrize("rnd_seed", range(128))
def test_high_labels_per_sample(rnd_seed):
    'reversed situation from the test "test_low_labels_per_sample"'

    seed = (rnd_seed + 11) * 129
    y_true = generate_random_labels(nsize=200, total_labels=10, min_num_labels=5, max_num_labels=10, seed=seed)
    y_pred = generate_random_labels(nsize=200, total_labels=10, min_num_labels=5, max_num_labels=10, seed=seed)

    f1 = f1_score(y_true, y_pred)
    p4 = p4_score(y_true, y_pred)
    
    assert f1.macro_average >= p4.macro_average
    assert f1.weighted_average >= p4.weighted_average
    assert f1.micro_average >= p4.micro_average
    assert f1.samples_average >= p4.samples_average


def test_multilabel_gradual_deterioration():
    rng = np.random.default_rng(seed=31415)
    dsize = 10000

    y_true = generate_random_labels(nsize=dsize, total_labels=10, min_num_labels=5, max_num_labels=10, seed=31415)
    y_pred = y_true.copy()

    metrics = p4_score(y_true, y_pred)

    # we should observe gradual deterioration of all the metrics
    # as we are adding random noise to y_pred
    for _ in range(10):
        indices = rng.integers(low=0, high=dsize, size=100)
        for i in indices:
            # divert all the labels at random position
            new_labels = [(k+1) % 10 for k in y_pred[i]]
            y_pred[i] = new_labels

        new_metrics = p4_score(y_true, y_pred)
        
        assert new_metrics.micro_average < metrics.micro_average
        assert new_metrics.macro_average < metrics.macro_average
        assert new_metrics.weighted_average < metrics.weighted_average
        assert new_metrics.samples_average < metrics.samples_average

        metrics = new_metrics