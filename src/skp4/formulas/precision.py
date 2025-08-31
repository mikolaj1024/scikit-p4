
from skp4.confusion import ConfusionQuartet


class PrecisionMixin:

    def formula(self, c: ConfusionQuartet):
        numerator = c.tp
        denominator = c.tp + c.fp
        return self.division(numerator, denominator)
    
