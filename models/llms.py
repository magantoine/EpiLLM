from decorators import expose
from .model_utils import Model
from transformers import (AutoModel,
                          AutoTokenizer,
                          AutoModelForCausalLM,
                          LlamaForCausalLM)
from huggingface_hub import snapshot_download
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os

if("OS_TYPE" not in os.environ):
    raise ValueError("Project is not properly setup (missing OS_TYPE in .env). Please run setup.sh")
if(os.environ["OS_TYPE"] == 'Linux'):
    vllm = __import__("vllm") ## conditional import
import torch

import hashlib
if("API_KEY" not in os.environ):
    print("Missing API key in .env file.")
    API_KEY = None
else :
    API_KEY = os.environ["API_KEY"]
    print("We've set API key : ", hashlib.md5(API_KEY.encode("utf-8")).hexdigest())
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
        if(API_KEY is not None):
            openai.api_key = API_KEY

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

    def set_key(self, apikey: str):
        openai.api_key = apikey
        self.api_key
    
    def see_api_key(self):
        return hashlib.md5(self.api_key)
hf_gen_params = [
    "temperature",
    "top_k",
    "top_p",
    "max_new_tokens",
]

ignore_gen_params = [
    "use_vllm"
]

@expose
class GenerationArg():
    
    def __init__(self,
                 temperature:float=1,
                 topk:int=1,
                 topp:float=1,
                 max_new_token:int=1,
                 presence_penalty:float=0.0,
                 frequency_penalty:float=1.0,
                 use_beam_search:bool=False,
                 logprobs:float=5,
                 best_of:int=1,
                 stop_seq:str=None,
                 use_vllm:bool=False
                 ) -> None:
        self.attr = {
            "temperature" : temperature,
            "top_k" : topk,
            "top_p" : topp,
            ("max_tokens" if use_vllm else "max_new_tokens") : max_new_token,
            "presence_penalty" : presence_penalty,
            "frequency_penalty" : frequency_penalty,
            "use_beam_search" : use_beam_search,
            "logprobs" : logprobs,
            "best_of" : best_of,
            "stop": stop_seq,
            "use_vllm": use_vllm
        }

        

        for n, v in self.attr.items():
            setattr(self, n, v)
        

        if(use_vllm):
            self.sampling_params = vllm.SamplingParams(**{arg: val for arg, val in self.attr.items() if arg not in ignore_gen_params})
        else:
            ## only HF params
            self.attr = {k: v for k, v in self.attr.items() if k in hf_gen_params}
        

    def __dict__(self):
        return self.attr
    
@expose
class HF_LLM(Model):

    def __init__(self,
                 model_name: str="epfl-llm/meditron-7b",
                 arg: GenerationArg=GenerationArg(),
                 device:str="",
                 use_vllm:bool=False,
                 lora_path:str|None=None) -> None:
        super().__init__()

        self.model_name = model_name
        self.arg = arg
        self.loaded = False
        self.device = device
        self.use_vllm = use_vllm
        self.use_lora = (lora_path is not None)
        
        lora_repo = snapshot_download(repo_id=lora_path) if self.use_lora else None
        self.lora_path = vllm.lora.request.LoRARequest("q&a_adapter", 1, lora_repo)

    def load(self) -> None:
        if(self.loaded):
            print("Model already loaded")
        else:
            if(not self.use_vllm):
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            else:
                self.client = vllm.LLM(
                    model=self.model_name,
                    tokenizer=self.model_name,
                    tensor_parallel_size=torch.cuda.device_count(),
                    enable_lora=self.use_lora
                )

            self.loaded = True
        
    def query(self, prompts:Union[List[str], str]) -> str:
        if(not self.loaded):
            raise ValueError("Cannot run since the model is not loaded")
        if(not self.use_vllm):
            gens = self.model.generate(
                **self.tok(
                    prompts, return_tensors="pt").to(self.device), 
                    pad_token_id=self.tok.eos_token_id,
                    **self.arg.attr) 
            return self.tok.batch_decode(gens)
        else :
            if(self.use_lora):
                return self.client.generate(
                    prompts,
                    sampling_params=self.arg.sampling_params,
                    lora_request=self.lora_path
                )
            else:
                return self.client.generate(
                    prompts,
                    sampling_params=self.arg.sampling_params
                )

    def set_arg(self,
                narg:GenerationArg) -> None:
        if(narg.use_vllm != self.arg.use_vllm):
            raise ValueError("Can't change the value of attribute use_vllm.")
        self.arg = narg
    