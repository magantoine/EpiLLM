import os

def init_ipynb():
    if(not os.getcwd().endswith("EpiLLM")):
        os.chdir("..")
    from dotenv import load_dotenv, find_dotenv
    ".env file found" if (envfound := load_dotenv(find_dotenv())) else ".env file not found"
    return envfound