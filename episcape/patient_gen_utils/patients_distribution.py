from decorators import expose

## imports 
import numpy as np
import pandas as pd
import os 
from random import randrange
from datetime import timedelta


BASE_PATH = os.environ["DIR_PATH"] if "DIR_PATH" in os.environ else "./"
DISTS_PATH = os.path.join(BASE_PATH, "episcape/patient_gen_utils/distributions")

"""
    - EPILEPSY_TYPE : brain region affected
    - EPILEPSY_FOCUS : focus/source of the epilepsy in the brain
    - SILENT : True/ False, Silent or Eloquant
    - COMORBIDITIES : does the patient has commorbidity
"""

def random_date(start, end):
    """
    This function will return a random datetime between two datetime 
    objects.
    """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = randrange(int_delta)
    return start + timedelta(seconds=random_second)

@expose
class ClinicalCondition:

    KEYS = [
        "EPILEPSY_TYPE",
        "EPILEPSY_FOCUS",
        "SILENT",
        "COMORBIDITIES",
        "FIRST_CRISIS",
        "LAST_CRISIS",
        "FREQUENCY",
        "INTENSITY",
        "DRUG_RESISTANT_EPILEPSY",
    ]
    EPILEPSY_TYPE_DIST = pd.read_csv(os.path.join(DISTS_PATH, "epilepsy_type.csv"))
    EPILEPSY_FOCUS_DIST = pd.read_csv(os.path.join(DISTS_PATH, "epilepsy_focus.csv"))
    
    def __init__(self, fix_inputs, patientdemos) -> None:
        self.fix_inputs = fix_inputs if fix_inputs is not None else {}
        self.attr = self.fix_inputs
        for key in ClinicalCondition.KEYS:
            if(key not in self.attr):
                self.attr[key] = ClinicalCondition.SAMPLE_FUNCS[key](self.attr, patientdemos)

    @staticmethod
    def sample_epilepsy_type(attr, patientdemos):
        dist = ClinicalCondition.EPILEPSY_TYPE_DIST
        return np.random.choice(dist.type, p=dist.p)
    
    @staticmethod
    def sample_epilepsy_focus(attr, patientdemos):
        dist = ClinicalCondition.EPILEPSY_FOCUS_DIST
        return np.random.choice(dist.focus, p=dist.p)
    
    @staticmethod
    def sample_epilepsy_silent(attr, patientdemos):
        return np.random.choice(["silent", "eloquent"], p=[1/2, 1/2])
    
    @staticmethod
    def sample_epilepsy_comorbidities(attr, patientdemos):
        comorbidities = [
            "obese",
            "cardio-vasculaire",
            "hemiplégique"
        ]
        p = 1 / len(comorbidities)
        return np.random.choice(comorbidities, p=[p] * len(comorbidities))
    
    @staticmethod
    def sample_first_crisis(attr, patientdemos):
        age = patientdemos.attr["age"]
        loc = age / 2
        scale = 0.3 * loc
        return int(np.random.normal(loc=loc, scale=scale, size=None))
    
    @staticmethod
    def sample_last_crisis(attr, patientdemos):
        dist = 1/np.arange(1, 20) / (1 / np.arange(1, 20)).sum()
        return int(np.random.choice(np.arange(1, 20), p=dist))
    
    @staticmethod
    def sample_frequency(attr, patientdemos):
        freq = [
            "every month",
            "every year",
            "every week"
        ]
        p = 1 / len(freq)
        return np.random.choice(freq, p=[p] * len(freq))

    @staticmethod
    def sample_intensity(attr, patientdemos):
        intens = [
            "very intense",
            "mildly intense",
            "not intense"
        ]
        p = 1 / len(intens)
        return np.random.choice(intens, p=[p] * len(intens))
    
    @staticmethod
    def sample_DRE(attr, patientdemos):
        return np.random.choice(["yes", "no"], p=[1/2, 1/2])


    SAMPLE_FUNCS = {
        "EPILEPSY_TYPE": sample_epilepsy_type,
        "EPILEPSY_FOCUS": sample_epilepsy_focus,
        "SILENT": sample_epilepsy_silent,
        "COMORBIDITIES": sample_epilepsy_comorbidities,
        "FIRST_CRISIS": sample_first_crisis,
        "LAST_CRISIS": sample_last_crisis,
        "FREQUENCY": sample_frequency,
        "INTENSITY": sample_intensity,
        "DRUG_RESISTANT_EPILEPSY" : sample_DRE,
    }

    
    def __str__(self) -> str:
        return str(self.attr)


demographics = pd.read_csv(os.path.join(DISTS_PATH,"demographics.csv"))

@expose
class Demographics:
    """
        Demographics of the patients. 
        Age, Gender, and Ethnic group should be present at
        equal rates in the training dataset
    """

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
        
    

        

    
