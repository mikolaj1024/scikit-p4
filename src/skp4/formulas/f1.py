 

from skp4.confusion import ConfusionQuartet

class F1Mixin:

    def formula(self, c: ConfusionQuartet):
        numerator = 2 * c.tp
        denominator = 2 * c.tp + c.fp + c.fn
        return self.division(numerator, denominator)
