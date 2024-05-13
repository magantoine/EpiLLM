from decorators import expose
from typing import (List, Callable, Any, Dict)
from models import Model
import json
from tqdm.notebook import tqdm
from pathlib import Path
import pickle

OFFSET_A = 65
letters = {i - OFFSET_A: chr(i) for i in range(OFFSET_A, OFFSET_A+26)}

@expose
class MCQBenchmark():


    def __init__(self,
                 path: str,
                 prompt_template: Callable[[str, Dict[str, str]], str],
                 cache_dir:str=None) -> None:
        self.path = path
        self.prompt_template = prompt_template
        ## stays None if no cache_dir indicated
        self.cache_dir = Path(cache_dir) if cache_dir is not None else cache_dir 
        
        with open(self.path, "r") as f: 
            self.mcq = json.load(f)

        MCQBenchmark.check_file(self.mcq)

    
    def assess(self,
               model: Model,
               cache_file: str=None):
        
        ## if cache_file indicated, the benchmark must have a cache_dir
        if(self.cache_dir is None and cache_file is not None):
            raise ValueError("No cache directory were indicated")
        
        cache_path = self.cache_dir / cache_file if cache_file is not None else None


        ret = self.mcq.copy()
        if(hasattr(model, "use_vllm") and model.use_vllm):
            ## process them all at once for vllm speedup
            res = model.query(
                [self.prompt_template(q["question"]) for q in ret]
            )

            MCQBenchmark.cache_res(res, cache_path) if cache_file is not None else None
            return res
        else :
            ## process them one by one
            for q in tqdm(ret):
                formatted_prompt = self.prompt_template(q["question"])
                answer = model.query(formatted_prompt)
                q["prediction"] = answer
        
        MCQBenchmark.cache_res(res, cache_path) if cache_file is not None else None
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
        

    @staticmethod
    def cache_res(res, cache_path):
        with open(cache_path, "wb") as cf:
            pickle.dump(res, cf)
        



    

    






