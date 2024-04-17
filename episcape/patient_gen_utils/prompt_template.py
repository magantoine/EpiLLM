from decorators import expose

import pandas as pd
import numpy as np
from typing import List, Dict
import random

import os
if("DIR_PATH" in os.environ):
    DIR = os.environ["DIR_PATH"]
    feats = ["label", "Synonyms", "Definitions"]
    epio = pd.read_csv(os.path.join(DIR, "docs/EPIO.csv"), usecols=feats).dropna()
    epio["Synonyms"] = epio["Synonyms"].dropna().apply(lambda x : x.replace("|", ", "))




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

            You'll answer my question one at a time, eahc of your answer will be a paragraph of the
            electronic record.

            1) Patient identy and cause of admission

            In this part you will describe the demographics of the patient. You should mention their age, 
            ethinicity, whether they are left or right handed, and their age. You should explained that Dr. X
            referred them recently for a complete pre surgical check up.

            2) Epilepsy history

            In this part, you will describe the epilepsy history of the patient. You should describe when the
            patient started having crisis, if they occur at night or during the day, the frequency and add a 
            medical, thorrow thorough description of the crisis making use of many neuroscientifc words like :

            {ontology}

            3) Medical History and behavioural history

            You will describe potential other risk factor that may appear in the patient's history. They 
            may of may not be relevant but you should mention them. Ranging from comorbidites, other desease,
            comorbidities, risk factors etc. 

            4) What investigation has been done

            List of the EEGs, video-EEGs, MRI, iSPECT.
            You should create a bullet point list with each examination done by the patient, if they went well
            and if they 

            5) Precendent medical trials

            What medicine (anti-epileptic has been tested) and what were the outcomes. If the patient is
            DRE then nothing has work (the epilepsy is drug-resistant).

            {drugs}

            6) Current medication

            Here you describe the current medication according to the previous paragraph. The medication should be
            {drugs[0]}

            7) Allergies and family history

            In this paragraph you describe briefly the family history (if the families have an precedent of epilepsy
            (there may not be any). You will also describe if the patient has any relevant alergies (it may not have any).

            8) Social background

            In this paragraph you will describe the socio demographic background of the person. What kind of scolarship they had
            where they live, whether their condition allows them to be autonomous.

        """,
        [
            """
                {add_info}
                You are a doctor in neurology. You'll be given the data from a patient's electronic record.
                You'll need to produce an electronic health record. It should be a long paragraph
                where you develop all of the person's medical journey. The patient has been diagnosed at some
                point with epilepsy.

                The given patient has :
                {add_info}

                You'll answer my question one at a time, eahc of your answer will be a paragraph of the
                electronic record.

                1) Patient identy and cause of admission

                In this part you will describe the demographics of the patient. You should mention their age, 
                ethinicity, whether they are left or right handed, and their age. You should explained that Dr. X
                referred them recently for a complete pre surgical check up.
            """,
            """
                {add_info}
                2) Epilepsy history

                In this part, you will describe the epilepsy history of the patient. You should describe when the
                patient started having crisis, if they occur at night or during the day, the frequency and add a 
                medical, thorrow thorough description of the crisis making use of many neuroscientifc words like :

                {ontology}
            """,
            """
                {add_info}
                3) Medical History and behavioural history

                You will describe potential other risk factor that may appear in the patient's history. They 
                may of may not be relevant but you should mention them. Ranging from comorbidites, other desease,
                comorbidities, risk factors etc. 
            """,
            """
                {add_info}
                4) What investigation has been done

                List of the EEGs, video-EEGs, MRI, iSPECT.
                You should create a bullet point list with each examination done by the patient, if they went well
                and if they 
            """,
            """
                {add_info}
                5) Precendent medical trials

                What medicine (anti-epileptic has been tested) and what were the outcomes. If the patient is
                DRE then nothing has work (the epilepsy is drug-resistant).

                {drugs}
            """,
            """
                6) Current medication

                Here you describe the current medication according to the previous paragraph. The medication should be
                {drugs[0]}
            """,
            """
                7) Allergies and family history

                In this paragraph you describe briefly the family history (if the families have an precedent of epilepsy
                (there may not be any). You will also describe if the patient has any relevant alergies (it may not have any).
            """,
            """
                8) Social background

                In this paragraph you will describe the socio demographic background of the person. What kind of scolarship they had
                where they live, whether their condition allows them to be autonomous.
            """
        ],
        """
            You are a doctor in neurology. You'll be given the data from a patient's electronic record.
            You'll need to produce an electronic health record. It should be a long paragraph
            where you develop all of the person's medical journey. The patient has been diagnosed at some
            point with epilepsy.

            The given patient has :
            {add_info[total]}

            You'll answer my question one at a time, eahc of your answer will be a paragraph of the
            electronic record.

            1) Patient identy and cause of admission

            In this part you will describe the demographics of the patient. You should mention their age, 
            ethinicity, whether they are left or right handed, and their age. You should explained that Dr. X
            referred them recently for a complete pre surgical check up.

            Recall that the patient's demographics are :
            {add_info[demos]}

            2) Epilepsy history

            In this part, you will describe the epilepsy history of the patient. You should describe when the
            patient started having crisis, if they occur at night or during the day, the frequency and add a 
            medical, thorrow thorough description of the crisis making use of many neuroscientifc words like :

            {ontology}

            3) Medical History and behavioural history

            You will describe potential other risk factor that may appear in the patient's history. They 
            may of may not be relevant but you should mention them. Ranging from comorbidites, other desease,
            comorbidities, risk factors etc. 

            Recall the patient's condition is given by :
            {add_info[medic]}

            4) What investigation has been done

            List of the EEGs, video-EEGs, MRI, iSPECT.
            You should create a bullet point list with each examination done by the patient, if they went well
            and if they 

            5) Precendent medical trials

            What medicine (anti-epileptic has been tested) and what were the outcomes. If the patient is
            DRE then nothing has work (the epilepsy is drug-resistant).

            {drugs}

            6) Current medication

            Here you describe the current medication according to the previous paragraph. The medication should be
            {drugs[0]}

            7) Allergies and family history

            In this paragraph you describe briefly the family history (if the families have an precedent of epilepsy
            (there may not be any). You will also describe if the patient has any relevant alergies (it may not have any).

            8) Social background

            In this paragraph you will describe the socio demographic background of the person. What kind of scolarship they had
            where they live, whether their condition allows them to be autonomous.

        """]
             
def get_prompt_type(pt):
    match str(type(pt)):
        case "<class 'list'>":
            return "COMPOUND"
        case "<class 'str'>":
            return "SIMPLE"
        case _:
            raise ValueError("Unknown prompt type, must be 'str' (SIMPLE) or 'list' (COMPOUND)")
        




def process_selected_drugs(select_drugs):
    return "\n".join(
        list(
            select_drugs.apply(
                lambda drug: f"- {eval(drug['molecules'])[0]} (e.g : {', '.join(eval(drug['brands']))})"
            , axis=1).values
        )
    )
    
def select_drugs():
    nb_drugs = random.randint(1, 5)
    drugs_sel = np.random.choice(np.arange(10), nb_drugs, replace=False)
    return pd.read_csv(DIR + "docs/drugs/drugs_top10.csv").iloc[drugs_sel]


def format_prompt(pt, add_info, ontology, pttype, drugs):
    if(pttype == "SIMPLE"):
        return pt.format(**{"add_info":add_info, "ontology":ontology, "drugs": drugs})
    if(pttype == "COMPOUND"):
        return [_.format(**{"add_info":add_info, "ontology":ontology, "drugs": drugs}) for _ in pt]
                        
def format_information(add_info: Dict[str, str]) -> str:
    return "\n".join(
            [
                f"\t\t- {str(k).replace('_', ' ').lower()}: {str(v)}" for k, v in add_info.items()
            ]
        )

INFO_CATEGORIES = {
    "demos": [
        "gender",
        "age",
        "ethnic_group"
    ],
    "medic": [
        "EPILEPSY_TYPE",
        'ethnic_group',
        'EPILEPSY_TYPE',
        'EPILEPSY_FOCUS',
        'SILENT',
        'COMORBIDITIES',
        'FIRST_CRISIS'
        'LAST_CRISIS',
        'FREQUENCY',
        'INTENSITY',
        'DRUG_RESISTANT_EPILEPSY',
        'a'
        'b',
        'c',
        'd',
        'e'
    ],
    "drugs": [
        "MEDICATION"
    ]
,}


@expose
def get_prompt(version=0,
               add_info=None,
               ontology=None) -> str:
    
    if(version < 3):
        drugs_dev = process_selected_drugs(select_drugs())
        add_info = format_information(add_info)
    elif(version == 4):
        ## This is the distillInfo version
        drugs_dev = process_selected_drugs(pd.DataFrame(add_info["MEDICATION"]))
        add_info = {
            "total": format_information(add_info),
            "demos": format_information({k: v for k, v in add_info.items() if k in INFO_CATEGORIES["demos"]}),
            "medic": format_information({k: v for k, v in add_info.items() if k in INFO_CATEGORIES["medic"]}),
            "drugs": format_information({k: v for k, v in add_info.items() if k in INFO_CATEGORIES["drugs"]})
        }
    
    
    ## ONTOLOGY
    ontology_dev = ""
    if(ontology):
        word_cnt = np.random.randint(5)
        for label, synonyms, definitions in epio.sample(word_cnt).values:
            ontology_dev += f"- {label} (synonyms : {synonyms}) : {definitions}\n"

    pttype = get_prompt_type(TEMPLATES[version])
    return (
        pttype, format_prompt(TEMPLATES[version], add_info, ontology_dev, pttype, drugs_dev)
    )



REFINE_TEMPLATE = """
"""

@expose
def get_refine_prompt_template(initial_info, first_gen):
    return REFINE_TEMPLATE.format(**{
        "initial_info": initial_info,
        "first_gen": first_gen
    })