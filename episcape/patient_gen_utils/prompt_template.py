from decorators import expose

import pandas as pd
import numpy as np

import os
if("DIR_PATH" in os.environ):
    DIR = os.environ["DIR_PATH"]
    feats = ["label", "Synonyms", "Definitions"]
    epio = pd.read_csv(os.path.join(DIR, "docs/EPIO.csv"), usecols=feats).dropna()
    epio["Synonyms"] = epio["Synonyms"].dropna().apply(lambda x : x.replace("|", ", "))


TEMPLATES = ["""
            You are a doctor in neurology. You'll be given the data from a patient's electronic record.
            You'll need to produce a dense version of the electronic health record. It should be a long paragraph
            where you develop all of the person's medical journey. The patient has been diagnosed at some
            point with epilepsy.

            The given patient has :
            {add_info}

            Give the dense electronic health record :
        """,
        """
            You are a doctor in neurology. You'll be given the data from a patient's electronic record.
            You'll need to produce a dense version of the electronic health record. It should be a long paragraph
            where you develop all of the person's medical journey. The patient has been diagnosed at some
            point with epilepsy.

            You should develop information as precisely as possible, using technical terms such as :

            {ontology}

            You need to write in a creative way, using creative sentences constructions.

            The given patient has :
            {add_info}

            Give the dense electronic health record :
        """]
             
@expose
def get_prompt(version=0,
               add_info=None,
               ontology=None) -> str:
    add_info = "\n".join(
        [
            f"\t\t- {str(k).replace('_', ' ').lower()}: {str(v)}" for k, v in add_info.items()
        ]
    )
    ontology_dev = ""
    if(ontology):
        word_cnt = np.random.randint(5)
        for label, synonyms, definitions in epio.sample(word_cnt).values:
            ontology_dev += f"- {label} (synonyms : {synonyms}) : {definitions}\n"
    return TEMPLATES[version].format(**{"add_info":add_info, "ontology":ontology_dev})
    
