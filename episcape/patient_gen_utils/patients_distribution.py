## imports 
import numpy as np
import pandas as pd
import os 
DISTS_PATH = "/Users/antoinemagron/Documents/EPFL/PDM/crchum/episcape/patient_gen_utils/distributions"

"""
    - EPILEPSY_TYPE : brain region affected
    - EPILEPSY_FOCUS : focus/source of the epilepsy in the brain
    - SILENT : True/ False, Silent or Eloquant
    - COMORBIDITIES : does the patient has commorbidity
"""


class ClinicalCondition:
    
    def __init__(self, fix_inputs) -> None:
        pass
    
    def __str__(self) -> str:
        return "\t> Clinical Cdt"

"""
    - EPILEPSY_TYPE : brain region affected
    - EPILEPSY_FOCUS : focus/source of the epilepsy in the brain
    - SILENT : True/ False, Silent or Eloquant
    - COMORBIDITIES : does the patient has commorbidity
"""


demographics = pd.read_csv(os.path.join(DISTS_PATH,"demographics.csv"))


class Demographics:

    def __init__(self, fix_inputs=None) -> None:
        self.fix_inputs = fix_inputs if fix_inputs is not None else {}
        self.gender = Demographics.sample_gender() if "gender" not in self.fix_inputs else self.fix_inputs["gender"]
        self.age = Demographics.sample_age(self.gender) if "age" not in self.fix_inputs else self.fix_inputs["age"]
        self.ethnic_group = Demographics.sample_ethnic_group() if "ethnic_group" not in self.fix_inputs else self.fix_inputs["ethnic_group"]
        
    @staticmethod
    def sample_age(gender):
        return np.random.choice(np.arange(len(demographics[gender])), p=(demographics[gender].values / demographics[gender].sum())) * 10 \
            + np.random.choice(np.arange(10))

    @staticmethod
    def sample_gender():
        p_male = demographics.Male.sum() / (demographics.Male.sum() + demographics.Female.sum())
        return np.random.choice(["Male", "Female"], p=[p_male, 1-p_male])
        
    @staticmethod
    def sample_ethnic_group():
        return "white"
        
    def __str__(self) -> str:
        return "\t> gender : " + str(self.gender) + "\n" \
         + "\t> age : " + str(self.age) + "\n" \
         + "\t> ethnic group : " + str(self.ethnic_group)

