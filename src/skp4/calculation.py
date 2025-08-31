import re
from skp4.binary import BinaryScore
from skp4.dataset import Dataset
from skp4.interface import ScoreInterface
from skp4.multilabel import MultiLabelScore


class ScoreCalculation:
    '''Factory class, creates proper inheritance chain for given formula and dataset.'''

    def __new__(cls, formula_class: type, dataset: Dataset , zero_division='warn') -> ScoreInterface:
        class_name = cls.__get_output_class__name(formula_class)

        if dataset.is_binary():
            output_class = type(class_name, (formula_class, BinaryScore), {})
        else:
            output_class = type(class_name, (formula_class, MultiLabelScore), {})

        return output_class(dataset, zero_division=zero_division)

    @classmethod
    def __get_output_class__name(cls, formula_class: type):
        first_word = re.match(r'^[A-Z][a-z,0-9]+', formula_class.__name__).group()
        return f'{first_word}Score'