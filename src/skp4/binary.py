

from skp4.confusion import ConfusionQuartet
from skp4.score import Score


class BinaryScore(Score):

    def calculate(self):
        cm = self.dataset.confusion_matrix()
        return self.formula(ConfusionQuartet.from_matrix(cm))

