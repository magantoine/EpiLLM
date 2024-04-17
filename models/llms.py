from decorators import expose
from .model_utils import Model
from transformers import (AutoModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          LlamaForCausalLM)


import os
if("API_KEY" not in os.environ):
    print("Missing API key in .env file.")
    API_KEY = None
else :
    API_KEY = os.environ["API_KEY"]
from typing import List, Dict, Union
import time

## OpenAI library
import openai
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    Timeout,
)


@expose
class EpiLLM(Model):
    def __init__(self) -> None:
        pass


@expose
class OpenAIGPT(Model):
    def __init__(self,
                 model: str,
                 temperature:float=0) -> None:
        super().__init__()
        self.model = model
        self.api_key = API_KEY
        self.temperature = temperature
        openai.api_key = API_KEY
        # print("LOADED API KEY : ", self.api_key)

    def query(self, 
              prompt: Dict[str, str]) -> None:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=prompt,
                request_timeout=20,
                temperature=self.temperature
            )
            return response["choices"][0]["message"]["content"]
        except (
            RateLimitError,
            ServiceUnavailableError,
            APIError,
            Timeout,
        ) as e:  # Exception
            print(f"Timed out {e}. Waiting for 10 seconds.")
            time.sleep(10)

@expose
class GenerationArg():
    
    def __init__(self,
                 temperature:float=1,
                 topk:int=1,
                 topp:float=1,
                 max_new_token:int=1
                 ) -> None:
        self.attr = {
            "temperature" : temperature,
            "top_k" : topk,
            "top_p" : topp,
            "max_new_tokens" : max_new_token
        }

        for n, v in self.attr.items():
            setattr(self, n, v)

    def __dict__(self):
        return self.attr
    
@expose
class HF_LLM(Model):

    def __init__(self,
                 model_name: str="epfl-llm/meditron-7b",
                 arg: GenerationArg=GenerationArg(),
                 device:str="") -> None:
        super().__init__()

        self.model_name = model_name
        self.arg = arg
        self.loaded = False
        self.device = device

    def load(self) -> None:
        if(self.loaded):
            print("Model already loaded")
        else:
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self.loaded = True

    def query(self, prompts:Union[List[str], str]) -> str:
        if(not self.loaded):
            raise ValueError("Cannot run since the model is not loaded")
        gens = self.model.generate(
            **self.tok(
                prompts, return_tensors="pt").to(self.device), 
                pad_token_id=self.tok.eos_token_id,
                **self.arg.attr) 
        return self.tok.batch_decode(gens)

    def set_arg(self,
                narg:GenerationArg) -> None:
        self.arg = narg
    


        