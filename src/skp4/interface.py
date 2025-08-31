
from abc import ABC, abstractmethod

import numpy as np

from skp4.confusion import ConfusionQuartet
from skp4.result import MultiResult


class ScoreInterface(ABC):

    @abstractmethod
    def calculate(self) -> np.number | MultiResult:
        ...

    @abstractmethod
    def division(self, numerator: np.number, denominator: np.number) -> np.number:
        ...

    @abstractmethod
    def formula(self, cq: ConfusionQuartet):
        ...
