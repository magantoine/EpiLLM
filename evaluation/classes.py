from decorators import expose
from typing import List
from .evaluate_utils import (SingleExperiment)

@expose
class GeneralExperiment(SingleExperiment):

    def __init__(self, preds: List, target: List) -> None:
        super().__init__(preds, target)

@expose
class SpecializedExperiment(SingleExperiment):

    def __init__(self, preds: List, target: List) -> None:
        super().__init__(preds, target)



    

    






