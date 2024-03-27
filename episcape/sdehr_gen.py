from decorators import expose


## super class
from .gen_utils import Generator
from .patient_gen import Patient
from .patient_gen_utils import get_prompt


## API key from .env file
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
class SDeHRGenerator(Generator):
    
    def __init__(self, temperature, version) -> None:
        self.api_key = API_KEY
        openai.api_key = API_KEY
        print("LOADED API KEY : ", self.api_key)
        self.temperature = temperature
        self.version = version


    def generate_sdehr(self, patient: Patient) -> str:
        prompt = get_prompt(add_info=dict(patient), version=self.version, ontology=self.version > 1)
        return self.query(messages=[
            {
                "role": "system",
                "content": "You are a doctor in neurology."
            },
            {
                "role": "user",
                "content": prompt
            }
        ], model="gpt-3.5-turbo")


    def query(self, 
            messages:List[Dict[str, str]],
            model: str="gpt-4"):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
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