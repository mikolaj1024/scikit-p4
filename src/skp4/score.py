import logging

import numpy as np

from skp4.dataset import Dataset
from skp4.exceptions import SklearnP4Exception
from skp4.interface import ScoreInterface


log = logging.getLogger(__name__)

class Score(ScoreInterface):

    def __init__(self, dataset: Dataset, zero_division='warn'):
        '''
        `zero_division`: {“warn”, 0.0, 1.0, np.nan}
        '''
        self.dataset = dataset
        self.zero_division = zero_division

    def division(self, numerator, denominator):
        if np.isscalar(denominator):
            return self.scalar_division(numerator, denominator)
        else:
            return self.array_division(numerator, denominator)

    def scalar_division(self, numerator, denominator):
        if denominator != 0:
            return numerator / denominator
        else:
            return self.get_default_zerodiv_value()
        

    def array_division(self, numerator, denominator):
        if not np.any(denominator == 0):
            return numerator / denominator
        
        zv = self.get_default_zerodiv_value()
        placeholder = np.full_like(numerator, zv, dtype=float)
        return np.divide(
            numerator, denominator, 
            out = placeholder, 
            where = (denominator != 0)
        )


    def get_default_zerodiv_value(self):
          match self.zero_division:
            case "warn":
                log.warning('division by zero during metric calculation')
                return 0.0
            case 0.0 | 1.0:
                return self.zero_division
            case _ if np.isnan(self.zero_division):
                return np.nan
            case _:
                raise SklearnP4Exception(f'unsupported zero_division value: {self.zero_division}')