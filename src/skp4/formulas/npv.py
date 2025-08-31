
from skp4.confusion import ConfusionQuartet


class NpvMixin:

    def formula(self, c: ConfusionQuartet):
        numerator = c.tn
        denominator = c.tn + c.fn
        return self.division(numerator, denominator)
