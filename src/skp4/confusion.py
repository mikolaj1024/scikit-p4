import numpy as np
from dataclasses import dataclass

from skp4.dataset import Dataset


@dataclass
class ConfusionQuartet:
    tp: int | np.ndarray
    tn: int | np.ndarray
    fp: int | np.ndarray
    fn: int | np.ndarray

    @classmethod
    def from_matrix(cls, confusion_matrix: np.ndarray) -> 'ConfusionQuartet':
        'Convert classic, binary confusion matrix as received from scikit-learn - into quartet class'
        tn, fp, fn, tp = confusion_matrix.ravel()
        return cls(tp=tp, tn=tn, fp=fp, fn=fn)
    
    @classmethod
    def as_sums_per_labels(cls, confusion_matrices: np.ndarray) -> 'ConfusionQuartet':
        '''Take a list of confusion matrices per label and return the quartet:
        each value calculated as a sum of values over all the labels.
        To be used for calculating micro-average'''
        tn, fp, fn, tp = confusion_matrices.sum(axis=0).ravel()
        return cls(tp=tp, tn=tn, fp=fp, fn=fn)

    @classmethod
    def as_sums_per_samples(cls, dataset: Dataset) -> 'ConfusionQuartet':
        'as sums per sample - suming over labels'

        yt = np.array(dataset.y_true_binarized, dtype=bool)
        yp = np.array(dataset.y_pred_binarized, dtype=bool)

        tp = np.logical_and(yt, yp)
        tn = np.logical_and(~yt, ~yp)
        fp = np.logical_and(~yt, yp)
        fn = np.logical_and(yt, ~yp)

        # for each sample, keep sum of value over all labels
        tp = np.sum(tp, axis=1)
        tn = np.sum(tn, axis=1)
        fp = np.sum(fp, axis=1)
        fn = np.sum(fn, axis=1)

        return cls(tp=tp, tn=tn, fp=fp, fn=fn)
       
    