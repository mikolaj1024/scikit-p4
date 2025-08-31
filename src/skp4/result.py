
from dataclasses import dataclass


@dataclass
class MultiResult:
    'Result for multi-label and multi-class classification assessment'
    micro_average: float
    macro_average: float
    weighted_average: float
    samples_average: float

    def __repr__(self):
        return f'''{'micro avg:':>13} {self.micro_average:0.4f}
{'macro avg:':>13} {self.macro_average:0.4f}
{'weighted avg:':>13} {self.weighted_average:0.4f}
{'samples avg:':>13} {self.samples_average:0.4f}
'''