# Scikit-P4: P4 Metric Calculation for Python
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/scikit-p4.svg)](https://pypi.org/project/scikit-p4/)

Scikit-P4 is a Python library for calculating the P4 metric [1][2] across binary, multiclass, and multilabel classification tasks. Its API closely follows that of Scikit-learn’s metrics (e.g., `f1_score`).

# Installation
```
pip install scikit-p4
```

# Usage
## Binary Case
Example for a binary classification problem.
```python
from skp4.metrics import p4_score
y_true = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
y_pred = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
p4_score(y_true, y_pred)
```
**Output**
```
np.float64(0.6956521739130435)
```

## Multiclass Case
Example with multiple classes.
```python
from skp4.metrics import p4_score
y_true = ['versicolor', 'versicolor', 'setosa', 'setosa', 'virginica', 'virginica', 'setosa']
y_pred = ['versicolor', 'setosa', 'setosa', 'virginica', 'setosa', 'virginica', 'setosa']
p4_score(y_true, y_pred)
```
**Output**

Returns a `MultiResult` object:
```
   micro avg: 0.6617
   macro avg: 0.6520
weighted avg: 0.6405
 samples avg: 0.5714
```

## Multilabel Case
Example with multilabel classification.
```python
from skp4.metrics import p4_score
y_true = [['A', 'B'], ['A', 'C'], ['B'], ['A', 'B', 'C']]
y_pred = [['A'], ['A', 'B'], [], ['A', 'B', 'C']]
p4_score(y_true, y_pred)
```
**Output**

Returns a `MultiResult` object:
```
   micro avg: 0.6522
   macro avg: 0.5758
weighted avg: 0.5568
 samples avg: 0.1667
```

# Function signature
```python
def p4_score(y_true, y_pred, *, zero_division='warn') -> np.number | MultiResult:
    ...
```
Calculates the P4 metric for binary, multiclass, and multilabel classification tasks.
The classifier type is detected automatically.

Parameters:
* `y_true` – Ground-truth labels.
* `y_pred` – Predicted labels.
* `zero_division` – Defines the behavior when division by zero occurs (compatible with Scikit-learn):
    * `0.0` – return 0 in case of division by zero
    * `1.0` – return 1 in case of division by zero
    * `np.nan` – return NaN in case of division by zero
    * `warn` (default) – return 0 and issue a warning

# Remarks
1. In the **multiclass** case, the *samples average* for the P4 metric equals *accuracy*, just like for the F1 metric.

2. In the **multilabel** case, it is possible to obtain a P4 result below 1.0 even when `y_true` is perfectly matched. 
   This occurs due to division-by-zero issues, which are common to many true-negative–dependent metrics.
   To avoid this:
   * Use the *micro average*, which does not exhibit this issue.
   * Set the parameter `zero_division=1.0` during calculation.


# References
[1] Wikipedia – [P4 metric](https://en.wikipedia.org/wiki/P4-metric)

[2] Sitarz, Mikołaj (2023). *Extending F1 Metric: A Probabilistic Approach*. 
    Advances in Artificial Intelligence and Machine Learning, 03(2), 1025–1038. 
    [arXiv:2210.11997](https://arxiv.org/abs/2210.11997)

# Contributing / License
Contributions are welcome! If you would like to report a bug, suggest an improvement, 
or contribute code, please open an issue or submit a pull request on GitHub.

This project is licensed under the **BSD 3-Clause License**.  
See the [LICENSE](LICENSE) file for details.
