import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

DIR_PATH = os.environ["DIR_PATH"]
GEN_PATH = Path(DIR_PATH) / "docs/generations_llm"
MCQS = os.listdir(GEN_PATH)

print(MCQS)

