from collections import namedtuple
import logging
import numpy as np

from skp4.classification_type import ClassificationType
from skp4.confusion import ConfusionQuartet
from skp4.dataset import Dataset
from skp4.result import MultiResult
from skp4.score import Score

log = logging.getLogger(__name__)

MacroAndWeighted = namedtuple('MacroAndWeighted', ['macro', 'weighted'])

class MultiLabelScore(Score):

    def calculate(self) -> MultiResult:
        macro, weighted = self.macro_and_weighted_average(self.dataset)
        micro = self.micro_average(self.dataset)
        samples = self.samples_average(self.dataset)

        result = MultiResult(
            micro_average=micro,
            macro_average=macro,
            weighted_average=weighted,
            samples_average=samples
        )
        self.show_warnings(result=result)
        return result


    def macro_and_weighted_average(self, dataset: Dataset) -> MacroAndWeighted:
        confusion_matrices = dataset.confusion_matrix()
        
        # calculating given metric formula per each label
        metric_per_label = [] 
        for cm in confusion_matrices:
            cq = ConfusionQuartet.from_matrix(cm)
            metric_value = self.formula(cq)
            metric_per_label.append(metric_value)

        support = np.sum(dataset.y_true_binarized, axis=0)

        return MacroAndWeighted(
            macro=np.average(metric_per_label),
            weighted=np.average(metric_per_label, weights=support)
        )
    

    def micro_average(self, dataset: Dataset) -> np.number:
        confusion_matrices = dataset.confusion_matrix()
        micro_quartet = ConfusionQuartet.as_sums_per_labels(confusion_matrices)
        # return metric formula based on the sums of tp/tn/fp/fn per labels
        return self.formula(micro_quartet)


    def samples_average(self, dataset: Dataset) -> np.number:
        samples_quarted = ConfusionQuartet.as_sums_per_samples(dataset)
        score_per_sample = self.formula(samples_quarted)
        return np.mean(score_per_sample)

        
    def show_warnings(self, result: MultiResult):
        'show warnings for tn-sensitive metrics'
        zero_division_warnings = ['warn', 0.0]
        warning_text = '''Metrics depending on TN can lead to to below-zero result event for perfect y_pred match 
for multilabel case, due to division-by-zero problem. 
One can set parameter `zero_division` = 1.0 or use micro average to avoid such a problem.'''
        
        if self.dataset.classification_type != ClassificationType.MULTI_LABEL:
            return
        
        if self.zero_division not in zero_division_warnings:
            return
        
        if np.isclose(result.micro_average, 1.0):
            if not (np.isclose(result.macro_average, 1.0) and\
               np.isclose(result.weighted_average, 1.0) and\
               np.isclose(result.samples_average, 1.0)):
                    log.warning(warning_text)
            
            
