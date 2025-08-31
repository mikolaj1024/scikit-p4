

from skp4.confusion import ConfusionQuartet


class SpecificityMixin:
    
    def formula(self, c: ConfusionQuartet):
        numerator = c.tn
        denominator = c.tn + c.fp
        return self.division(numerator, denominator)
       