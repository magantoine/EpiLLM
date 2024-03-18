from typing import List
from .evaluate_utils import (SingleExperiment)


class GeneralExperiment(SingleExperiment):

    def __init__(self, preds: List, target: List) -> None:
        super().__init__(preds, target)


class SpecializedExperiment(SingleExperiment):

    def __init__(self, preds: List, target: List) -> None:
        super().__init__(preds, target)



    

    






