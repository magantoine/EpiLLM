from typing import (List,
                    Dict,
                    Any)
import pandas as pd
import numpy as np

## local imports
from .gen_utils import Generator
from .patient_gen_utils.patients_distribution import (
    Demographics,
    ClinicalCondition
)
from decorators import expose



@expose
class Patient:
    def __init__(self, fix_inputs) -> None:
        self.demos = Demographics(fix_inputs)
        self.clinical_cdt = ClinicalCondition(fix_inputs, self.demos)

    def __dict__(self) -> dict:
        return self.demos.attr | self.clinical_cdt.attr
    def __str__(self) -> str:
        return "> demographics : \n" + str(self.demos) + "\n" + "> clinical condition :" + str(self.clinical_cdt)
    

@expose
class PatientGenerator(Generator): 

    NOISE = [
        ("a", 1/3),
        ("b", 1/6),
        ("c", 1/2),
        ("d", 4/5),
        ("e", 8/9)
    ]
    def __init__(self) -> None:
        pass

    def generate(self,
                 n:int=100) -> pd.DataFrame:
        patients_df = pd.DataFrame(
            [Patient(None).__dict__() for _ in range(n)]
        )
        return self.add_noise(patients_df)


    def add_noise(self, df):
        for noise, p in PatientGenerator.NOISE:
            df[noise] = np.random.choice([True, False], size=len(df.index), p=[p, 1-p])
        return df


