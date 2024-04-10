from decorators import expose
from typing import (List, Callable, Any, Dict)
from models import Model
import json

OFFSET_A = 65
letters = {i - OFFSET_A: chr(i) for i in range(OFFSET_A, OFFSET_A+26)}

@expose
class MCQBenchmark():


    def __init__(self,
                 path: str,
                 prompt_template: Callable[[str, Dict[str, str]], str]) -> None:
        self.path = path
        self.prompt_template = prompt_template
        
        with open(self.path, "r") as f: 
            self.mcq = json.load(f)

        MCQBenchmark.check_file(self.mcq)

    
    def assess(self,
               model: Model):
        ret = self.mcq.copy()
        for q in ret:
            formatted_prompt = self.prompt_template(q["question"])
            answer = model.query(formatted_prompt)
            q["prediction"] = answer
        return ret

    @staticmethod
    def check_file(mcq):
        if(type(mcq) != list and any(type(_) != dict for _ in mcq)):
            raise ValueError("MCQ should be entered as a list of record")
        
        if(any(
            "question" not in q or "answer" not in q for q in mcq
            )
        ):
            raise ValueError("MCQ question should have a question and an answer")
        



    

    






