from typing import (
    List
)

class Experiment:

    def __init__(self) -> None:
        raise NotImplementedError()

    def result(self) -> float:
        raise NotImplementedError()
    

class SingleExperiment:

    def __init__(self,
                 preds: List,
                 target: List) -> None:
        pass

    def result(self) -> float:
        pass

class FullExperiment:

    def __init__(self,
                 exps: List[Experiment]) -> None:
        pass

    def result(self) -> float:
        pass
