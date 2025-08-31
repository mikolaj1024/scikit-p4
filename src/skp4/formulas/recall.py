
from skp4.confusion import ConfusionQuartet


class RecallMixin:

    def formula(self, c: ConfusionQuartet):
        numerator = c.tp
        denominator = c.tp + c.fn
        return self.division(numerator, denominator)
    
