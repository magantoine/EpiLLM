from decorators import expose

import pandas as pd
import numpy as np
from typing import List

import os
if("DIR_PATH" in os.environ):
    DIR = os.environ["DIR_PATH"]
    feats = ["label", "Synonyms", "Definitions"]
    epio = pd.read_csv(os.path.join(DIR, "docs/EPIO.csv"), usecols=feats).dropna()
    epio["Synonyms"] = epio["Synonyms"].dropna().apply(lambda x : x.replace("|", ", "))



"""

PROTO CANVA USED V2 :
- Description of the patient
- when did he start coming to the hostpital ? what for ?
- what tests were made ? with what result ?
- how is his medical journey ?
- How is he feeling about the surgery solution when he's been introduced to it ?
"""

TEMPLATES = [
        """
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
        """,
        """
            You are a doctor in neurology. You'll be given the data from a patient's electronic record.
            You'll need to produce an electronic health record. It should be a long paragraph
            where you develop all of the person's medical journey. The patient has been diagnosed at some
            point with epilepsy.

            The given patient has :
            {add_info}

            You should answer in one paragraph per each of these paragraphs :
            - Description of the patient
            - when did he start coming to the hostpital ? what for ?
            - what tests were made ? with what result ?
            - how is his medical journey ?
            - How is he feeling about the surgery solution when he's been introduced to it ?

            Each paragraph should be separated by a skipped line and a line for the title of the paragraph.
            The title could be a rephrasing of the question.

            Give the dense electronic health record :
        """,
        [
            """
                You are a doctor in neurology. You'll be given the data from a patient's electronic record.
                You'll need to produce an electronic health record. It should be a long paragraph
                where you develop all of the person's medical journey. The patient has been diagnosed at some
                point with epilepsy.

                The given patient has :
                {add_info}

                You'll answer my question one at a time, eahc of your answer will be a paragraph of the
                electronic record.

                1) Describe the patient
            """,
            """
                2) when did he start coming to the hostpital ? what for ?
            """,
            """
                3) what tests were made ? with what result ?
            """,
            """
                4) how is his medical journey ?
            """,
            """
                5) How is he feeling about the surgery solution when he's been introduced to it ?
            """,


        ]]
             
def get_prompt_type(pt):
    match str(type(pt)):
        case "<class 'list'>":
            return "COMPOUND"
        case "<class 'str'>":
            return "SIMPLE"
        case _:
            raise ValueError("Unknown prompt type, must be 'str' (SIMPLE) or 'list' (COMPOUND)")
        
def format_prompt(pt, add_info, ontology, pttype):
    if(pttype == "SIMPLE"):
        return pt.format(**{"add_info":add_info, "ontology":ontology})
    if(pttype == "COMPOUND"):
        return [_.format(**{"add_info":add_info, "ontology":ontology}) for _ in pt]
                        
                         

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
    pttype = get_prompt_type(TEMPLATES[version])
    return (
        pttype, format_prompt(TEMPLATES[version], add_info, ontology_dev, pttype)
    )
    
