
from skp4.confusion import ConfusionQuartet


# TODO remove
class FluftuMetricMixin:
    'Not a real metric - a testing scenarios helper'

    def formula(self, c: ConfusionQuartet):
        return 1000 * c.tp + 100 * c.tn + 10 * c.fp + c.fn
    


class ZenMetricMixin:
    'Not a real metric - a testing scenarios helper'

    def formula(self, c: ConfusionQuartet):
        return self.division(c.tp, c.tn)
    
