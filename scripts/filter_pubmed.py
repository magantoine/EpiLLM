from pathlib import Path
import datasets
from typing import Dict, Any
from time import time

DOC_PATH = Path("../docs")
ds = datasets.load_dataset("pubmed", num_proc=1)
FLAG = "epilepsy"


def filter(pub: Dict[Any, Any]):
    first_pub_year = pub["PubmedData"]["History"]["PubMedPubDate"]["Year"][0]
    article = pub["MedlineCitation"]["Article"]
    abstract = article["Abstract"]["AbstractText"]
    title = article["ArticleTitle"]
    return first_pub_year >= 2000 and ((FLAG in title or FLAG in abstract) and abstract != '')


start = time()
ds = ds.filter(filter, num_proc=1)


ds.save_to_disk("../docs/pubmed_saved.hf")
