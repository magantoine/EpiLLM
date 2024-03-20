## local imports
from .gen_utils import Generator
from .patient_gen_utils.patients_distribution import (Demographics, ClinicalCondition)
from decorators import expose


## typing
from typing import (List,
                    Dict,
                    Any)

## aux
import pandas as pd


@expose
class Patient:
    def __init__(self, fix_inputs) -> None:
        self.demos = Demographics(fix_inputs)
        self.clinical_cdt = ClinicalCondition(fix_inputs)

    def __dict__(self) -> dict:
        return self.demos.attr | self.clinical_cdt.attr
    def __str__(self) -> str:
        return "> demographics : \n" + str(self.demos) + "\n" + "> clinical condition :" + str(self.clinical_cdt)
    
@expose
class PatientGenerator(Generator): 

    def __init__(self) -> None:
        """
        """
        pass

    def generate(self,
                 n:int=100,
                 return_type:type=List) -> List[Patient]:
        if(return_type == List):
            return [Patient(None) for _ in range(n)]
        elif(return_type == pd.DataFrame):
            return pd.DataFrame(
                [Patient(None).__dict__() for _ in range(n)]
            )

