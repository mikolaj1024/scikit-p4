

from skp4.confusion import ConfusionQuartet


class P4Mixin:
    
    def formula(self, c: ConfusionQuartet):
        numerator = 4 * c.tp * c.tn
        denominator = 4 * c.tp * c.tn + (c.tp + c.tn) * (c.fp + c.fn)
        return self.division(numerator, denominator)


   
