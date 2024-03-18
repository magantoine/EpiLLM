## local imports
from .gen_utils import Generator
from .patient_gen_utils.patients_distribution import (Demographics, ClinicalCondition)


## typing
from typing import (List,
                    Dict,
                    Any)



class Patient:
    def __init__(self, fix_inputs) -> None:
        self.demos = Demographics(fix_inputs)
        self.clinical_cdt = ClinicalCondition(fix_inputs)

    def __str__(self) -> str:
        return "> demographics : \n" + str(self.demos) + "\n" + "> clinical condition : \n" + str(self.clinical_cdt)


class PatientGenerator(Generator): 

    def __init__(self) -> None:
        """
        """
        pass

    def generate(self, n:int=100) -> List[Patient]:
        return [Patient(None) for i in range(n)]

