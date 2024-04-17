from decorators import expose
from .patient_gen import PatientGenerator
from .sdehr_gen import SDeHRGenerator

from typing import List

@expose
class PipelineGenerator(): 

    def __init__(self,
                 temperature:float=0,
                 version:int=0) -> None:
        self.patient_gen = PatientGenerator()
        self.sdehr_gen = SDeHRGenerator(temperature, version)
    
    def generate(self, n:int=10) -> List[str] :
        test = self.patient_gen.generate(10)
        test["sdehr"] = [
            self.sdehr_gen.generate_sdehr(test_patient)
            for test_patient in test.to_dict(orient="records")
        ]
        return test