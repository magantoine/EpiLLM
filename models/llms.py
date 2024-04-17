from decorators import expose
from .model_utils import Model


import os
if("API_KEY" not in os.environ):
    print("Missing API key in .env file.")
    API_KEY = None
else :
    API_KEY = os.environ["API_KEY"]
from typing import List, Dict
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




        