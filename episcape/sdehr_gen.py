## super class
from .gen_utils import Generator

## API key from .env file
import os
if("API_KEY" not in os.environ):
    raise NotImplementedError("Missing API key in .env file.")
API_KEY = os.environ["API_KEY"]

## OpenAI library
import openai
from openai.error import (
    RateLimitError,
    ServiceUnavailableError,
    APIError,
    Timeout,
)



class SDeHRGenerator(Generator):
    
    def __init__(self) -> None:
        self.api_key = API_KEY
        openai.api_key = API_KEY
        print("LOADED API KEY : ", self.api_key)