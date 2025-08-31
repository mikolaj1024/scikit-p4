
import numpy as np

'Testing purpose binary dataset generator'

def generate_set(n=100000, positives_fraction=0.0005, tpr=0.95, tnr=0.95, seed=314151):
    '''Generate pseudorandom set of y_true and y_pred 
    having given properties:
    `n` - size of the set
    `positives_fraction` - fraction of positive samples in population
    `tpr` - true positive rate
    `tnr` - true negative rate
    `seed` - pseudorandom numbers generator seed
    '''

    n_positives = int(positives_fraction * n)
    n_negatives = n - n_positives

    tp = int(n_positives * tpr)
    fn = int(n_positives - tp)
    tn = int(n_negatives * tnr)
    fp = int(n_negatives - tn)
    return construct_set(tp, tn, fp, fn, seed=seed)
    


def construct_set(tp, tn, fp, fn, shuffle=True, seed=314151):
    '''Construct set of y_true and y_pred having exact number of samples:
    * `tp` - number of true positives
    * `tn` - number of true negatives
    * `fp` - number of false positives
    * `fn` - number of false negatives

    if `suffle` is true - samples are randomly shuffled
    '''
    # generate positives detected as positives
    p_as_p = np.full((tp, 2), [1, 1])

    # generate positives detected as negatives
    p_as_n = np.full((fn, 2), [1, 0])

    # generate negatives detected as negatives
    n_as_n = np.full((tn, 2), [0, 0])

    # generate negatives detected as positives
    n_as_p = np.full((fp, 2), [0, 1])

    a = np.concatenate((p_as_p, p_as_n, n_as_n, n_as_p))
    rnd = np.random.default_rng(seed)
    if shuffle:
        rnd.shuffle(a)

    y_true = a[:, 0]
    y_pred = a[:, 1]
    return y_true, y_pred
    
