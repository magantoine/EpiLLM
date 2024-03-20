from decorators import expose

## imports 
import numpy as np
import pandas as pd
import os 
BASE_PATH = os.environ["DIR_PATH"] if "DIR_PATH" in os.environ else "./"
DISTS_PATH = os.path.join(BASE_PATH, "episcape/patient_gen_utils/distributions")

"""
    - EPILEPSY_TYPE : brain region affected
    - EPILEPSY_FOCUS : focus/source of the epilepsy in the brain
    - SILENT : True/ False, Silent or Eloquant
    - COMORBIDITIES : does the patient has commorbidity
"""

@expose
class ClinicalCondition:

    KEYS = ["EPILEPSY_TYPE", "EPILEPSY_FOCUS", "SILENT", "COMORBIDITIES"]
    EPILEPSY_TYPE_DIST = pd.read_csv(os.path.join(DISTS_PATH, "epilepsy_type.csv"))
    EPILEPSY_FOCUS_DIST = pd.read_csv(os.path.join(DISTS_PATH, "epilepsy_focus.csv"))
    
    def __init__(self, fix_inputs) -> None:
        self.fix_inputs = fix_inputs if fix_inputs is not None else {}
        self.attr = self.fix_inputs
        for key in ClinicalCondition.KEYS:
            if(key not in self.attr):
                self.attr[key] = ClinicalCondition.SAMPLE_FUNCS[key](self.attr)

    @staticmethod
    def sample_epilepsy_type(attr):
        dist = ClinicalCondition.EPILEPSY_TYPE_DIST
        return np.random.choice(dist.type, p=dist.p)
    
    @staticmethod
    def sample_epilepsy_focus(attr):
        dist = ClinicalCondition.EPILEPSY_FOCUS_DIST
        return np.random.choice(dist.focus, p=dist.p)
    
    @staticmethod
    def sample_epilepsy_silent(attr):
        return np.random.choice(["silent", "eloquent"], p=[1/2, 1/2])
    
    @staticmethod
    def sample_epilepsy_comorbidities(attr):
        comorbidities = [
            "obese",
            "cardio-vasculaire",
            "hemiplégique"
        ]
        p = 1 / len(comorbidities)
        return np.random.choice(comorbidities, p=[p] * len(comorbidities))

    SAMPLE_FUNCS = {
        "EPILEPSY_TYPE": sample_epilepsy_type,
        "EPILEPSY_FOCUS": sample_epilepsy_focus,
        "SILENT": sample_epilepsy_silent,
        "COMORBIDITIES": sample_epilepsy_comorbidities,
    }

    
    def __str__(self) -> str:
        return str(self.attr)


demographics = pd.read_csv(os.path.join(DISTS_PATH,"demographics.csv"))

@expose
class Demographics:

    KEYS = ["gender", "age", "ethnic_group"]

    def __init__(self, fix_inputs=None) -> None:
        self.fix_inputs = fix_inputs if fix_inputs is not None else {}
        self.attr = self.fix_inputs
        for key in Demographics.KEYS:
            if(key not in self.attr):
                self.attr[key] = Demographics.SAMPLES_FUNCS[key](self.attr)
        


    @staticmethod
    def sample_age(attr):
        if("gender" not in attr):
            raise ValueError("Need the patient's gender to sample the age distribution.")
        gender = attr["gender"]
        return np.random.choice(np.arange(len(demographics[gender])), p=(demographics[gender].values / demographics[gender].sum())) * 10 \
            + np.random.choice(np.arange(10))

    @staticmethod
    def sample_gender(attr):
        p_male = demographics.Male.sum() / (demographics.Male.sum() + demographics.Female.sum())
        return np.random.choice(["Male", "Female"], p=[p_male, 1-p_male])
        
    @staticmethod
    def sample_ethnic_group(attr):
        return "white"
        
    SAMPLES_FUNCS = {
        "gender": sample_gender,
        "age": sample_age,
        "ethnic_group": sample_ethnic_group
    }

    def __str__(self) -> str:
        return str(self.attr)

