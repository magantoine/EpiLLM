from decorators import expose
from typing import (List, Callable, Any, Dict)
from models import Model
import json
from tqdm.notebook import tqdm
from pathlib import Path
import pickle
from models.qa_prompts import QA_PROMPTS
from .offline_metrics import embSim, embed_dataset
import numpy as np




from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
from pathlib import Path
import random
if("DIR_PATH" not in os.environ):
    raise ValueError("Project is not properly setup (missing DIR_PATH in .env). Please run setup.sh")
else :
    DIR_PATH = Path(os.environ["DIR_PATH"])

OFFSET_A = 65
letters = {i - OFFSET_A: chr(i) for i in range(OFFSET_A, OFFSET_A+26)}

with open(DIR_PATH / "docs/support_set.json", "rb") as f:
    SUPPORT_SET = pickle.load(f)



@expose
class MCQBenchmark():


    def __init__(self,
                 path: str,
                 prompt_template: Callable[[str, Dict[str, str]], str],
                 cache_dir:str=None,
                 support_type:str=None,
                 n_shots:int=3) -> None:
        
        """
            /!\ support_type=None is kept for retrocompatility /!\
            
            CASE support_type in ["deterministic", None]:
                args :
                    - question (with field "question" and "answer")
            CASE support_type in ["shots", ""]:
                args :
                    - question
                    - shots
            
        """
        if(support_type not in [None, "deterministic", "random", "kNN"]):
            raise ValueError("support_type must be one of : 'deterministic', 'random', 'kNN'")
        
        self.path = path
        self.prompt_template = prompt_template 
        self.support_type = support_type
        self.n_shots = n_shots
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
        all_shots = self.get_shot_list()
        print(all_shots)

        if(hasattr(model, "use_vllm") and model.use_vllm):
            ## process them all at once for vllm speedup
            res = model.query(
                [self.prompt_template(q["question"], shots) for q, shots in zip(ret, all_shots)]
            ) if self.support_type != None else model.query(
                [self.prompt_template(q["question"]) for q in ret]
            ) ## RETRO

            MCQBenchmark.cache_res(res, cache_path) if cache_file is not None else None
            return res
        else :
            ## process them one by one
            for q, shots in tqdm(list(zip(ret, all_shots))):
                formatted_prompt = self.prompt_template(q["question"], shots) if self.support_type != None else self.prompt_template(q["question"]) ## RETRO
                answer = model.query(formatted_prompt)
                q["prediction"] = answer
        
        MCQBenchmark.cache_res(res, cache_path) if cache_file is not None else None
        return ret


    def get_shot_list(self) -> List[List[str]]:
        def get_question_shot(q):
            if(self.support_type == None):
                return
            if(self.support_type == "deterministic"):
                ## THE SHOTS ARE IN THE CLASSICAL PROMPT
                return QA_PROMPTS["1cot_answer_align"]["shots"]
            if(self.support_type == "random"):
                ## SELECT FOR EACH SAMPLE A SET OF N_SHOTS SHOTS IN THE SUPPORT SET
                return [_["shot"] for _ in random.sample(SUPPORT_SET, self.n_shots)]
            if(self.support_type == "kNN"):
                ## SELECT THE N_SHOTS MOST SIMILAR SHOTS IN THE SUPPORT SET
                sims = embSim([_["embedding"] for _ in SUPPORT_SET] + [embed_dataset(q["question"]).detach().squeeze(0)], in_type="pt")[-1]
                kNN_index = np.argsort(sims)[-self.n_shots:]
                return [_["shot"] for i, _ in enumerate(SUPPORT_SET) if i in kNN_index]
        return [get_question_shot(q) for q in self.mcq]
    
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
        



    

    






