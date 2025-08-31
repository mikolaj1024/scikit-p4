import numpy as np

from skp4.exceptions import SklearnP4Exception

def generate_set_from_confusion_matrix(cm: np.ndarray, seed=314151) -> tuple[np.ndarray, np.ndarray]:
    '''Generate `y_true` and `y_pred` from given confusion matrix
    Example:

```text
                Predicted
                A    B    C
Actual    A   [85    3    2]   <- 90 actual A samples
          B   [ 5   78    7]   <- 90 actual B samples  
          C   [ 2    4   84]   <- 90 actual C samples
```

    '''
    

    if cm.ndim != 2:
        raise SklearnP4Exception('confusion matrix should have 2 dimensions')
    
    rows, cols = cm.shape
    if rows != cols:
        raise SklearnP4Exception('confusion matrix should be square')

    n_classes = cm.shape[0]
    y_true = []
    y_pred = []

    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            count = cm[true_class, pred_class]
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_true)

    rng = np.random.default_rng(seed=seed)
    rng.shuffle(y_pred)

    return y_true, y_pred
