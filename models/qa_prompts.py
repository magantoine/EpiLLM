QA_PROMPTS = {
    "direct_answer_align" : {
        "system": "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.",
        "shots": [ ## 3 samples
            """-user: 
                Question :  A 63-year-old woman is hospitalized with herpes simplex virus encephalitis. Which of the following EEG patterns is most likely to be seen in this patient?
                    A. burst suppression
                    B. generalized periodic discharges
                    C. periodic lateralizing discharges
                    D. small sharp spikes
                    E. stimulus-induced, rhythmic, periodic, or ictal discharges
                -assistant:
                Answer : C<|STOP|>"""
                ,
                """-user:
                Question :  Which of the following EEG patterns is rarely seen in normal patients and has a high specificity for temporal lobe epilepsy?
                    A. 14 and 6 positive spikes
                    B. diffuse slow activity
                    C. rhythmic temporal delta
                    D. triphasic waves
                    E. wicket spikes
                -assistant:
                Answer : C<|STOP|>"""
                ,
                """-user:
                Question :  In addition to a risk of major congenital malformations, which of the following antiepileptic drugs is also associated with an elevated risk of autism spectrum disorders and lower IQ at age 6?
                    A. carbamazepine
                    B. lamotrigine
                    C. phenobarbital
                    D. phenytoin
                    E. valproate
                -assistant:
                Answer : E<|STOP|>"""
            ],
            "q_form": "-user\nQuestion : {q}\n-assistant:\nAnswer : "
    },
    "1cot_answer_align": {
        "system": """You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.
            Let's think step by step. You will end by telling the letter of the corresponding answer.
            """,
        "shots": [
            """-user: 
            Question :  A 63-year-old woman is hospitalized with herpes simplex virus encephalitis. Which of the following EEG patterns is most likely to be seen in this patient?
                A. burst suppression
                B. generalized periodic discharges
                C. periodic lateralizing discharges
                D. small sharp spikes
                E. stimulus-induced, rhythmic, periodic, or ictal discharges
            -assistant:
            Answer :  Periodic lateralizing discharges are EEG abnormalities commonly seen in certain scenarios, including herpes encephalitis. Other conditions in which periodic lateralizing discharges are seen include focal brain lesions, nonketotic hyperglycemia, alcohol withdrawal, and theophylline exposure. Generalized periodic discharges are more commonly seen in anoxic brain injury, Creutzfeldt-Jakob disease, and other diffuse processes. Stimulus-induced, rhythmic, periodic, or ictal discharges (SIRPIDs) are most commonly seen in critically ill patients, even those without neurologic illness. Burst suppression is typically seen in anoxic encephalopathy or in the setting of certain medications (eg, propofol, barbiturates).
            Therefore, the correct answer is  C.<|STOP|>"""
            ,
            """-user:
            Question :  Which of the following EEG patterns is rarely seen in normal patients and has a high specificity for temporal lobe epilepsy?
                A. 14 and 6 positive spikes
                B. diffuse slow activity
                C. rhythmic temporal delta
                D. triphasic waves
                E. wicket spikes
            -assistant:
            Answer :  Of the options listed above, rhythmic temporal delta activity is most suggestive of temporal lobe epilepsy. While sensitivity for this finding is not especially high, it is rarely seen outside of the context of temporal lobe seizures and therefore carries high specificity. Triphasic waves are most commonly seen in patients with metabolic encephalopathy. Diffuse slow activity is seen in patients with traumatic brain injury, subarachnoid hemorrhage, toxic-metabolic encephalopathies, and other processes that affect the brain more diffusely. Wicket spikes and 14 and 6 positive spikes are considered normal EEG variants seen in older adults and adolescents, respectively.
            Therefore, the correct answer is  C.<|STOP|>"""
            ,
            """-user:
            Question :  In addition to a risk of major congenital malformations, which of the following antiepileptic drugs is also associated with an elevated risk of autism spectrum disorders and lower IQ at age 6?
                A. carbamazepine
                B. lamotrigine
                C. phenobarbital
                D. phenytoin
                E. valproate
            -assistant:
            Answer :  Women treated with valproate (either monotherapy or in combination with other antiepileptics) are at elevated risk of giving birth to children with major congenital malformations. In addition, of the medications listed above, valproate is also associated with a higher risk of neurodevelopmental effects in children exposed during pregnancy, as measured by lower IQ at age 6 and a higher risk of autism spectrum disorders.
            Therefore, the correct answer is  E.<|STOP|>"""
        ],
        "q_form": "-user\nQuestion : {q}\n-assistant:\nAnswer : "
        

    },
    "cot_answer_align":{
        "system": """You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.
            Let's think step by step. You will end by telling the letter of the corresponding answer.
            """,
        "shots": [ ## 3 samples
            """-user:
            Question : A 38-year-old man undergoes routine EEG testing after having a first-time seizure. During photic stimulation, the EEG shows time locked bi-frontally predominant myogenic activity which increases in amplitude as the frequency of stimulation is increased. Which of the following is most associated with this finding?
                    A. An underlying predisposition to generalized seizures
                    B. A high state of anxiety or substance withdrawal
                    C. An increased risk for provoked seizures
                    D. A photochemical reaction of the light and EEG electrodes
            -assistant:
            Answer : - EEG testing on a 38-year-old man after his first seizure reveals a distinct pattern during photic stimulation: time-locked bi-frontally predominant myogenic activity.
                - This myogenic activity signifies muscle-related responses triggered by the light stimulation, with the amplitude increasing as the stimulation frequency rises.
                - Considering possible associations, option A, which suggests an underlying predisposition to generalized seizures, appears most relevant given the observed EEG pattern.
                - Options B and C, relating to anxiety, substance withdrawal, or provoked seizures, seem less directly connected to the specific muscle-related response observed during EEG testing.
                - Option D introduces a different phenomenon, a photochemical reaction, which doesn't align with the observed myogenic activity during photic stimulation. Therefore, option A is the most suitable interpretation of the findings.
                Therefore the correct answer is A.<|STOP|>""",
            """-user:
            Question : Surgery for which of the following types of epilepsy is least likely to result in seizure freedom?
                A. epilepsy due to a hemispheric syndrome
                B. epilepsy due to neonatal infarct
                C. epilepsy due to tumor
                D. mesial temporal lobe epilepsy
                E. nonlesional neocortical epilepsy
            -assistant:
            Answer : - The question presents various types of epilepsy and asks about the likelihood of seizure freedom after surgery for each type.
                - Options A, B, C, and D represent different types of epilepsy with identifiable lesions or specific areas of the brain affected
                - Option E, nonlesional neocortical epilepsy, stands out as it refers to epilepsy without a detectable lesion in the neocortex, which  makes surgical intervention more challenging.
                - The answer explanation emphasizes that the best surgical outcomes are typically seen in cases where there are clear lesions or identifiable epileptic foci, such as mesial temporal sclerosis, tumors, or hemispheric syndromes.
                - Consequently, option E is identified as the least likely to result in seizure freedom after surgery due to the absence of a detectable lesion, making it more difficult to target the epileptic focus accurately.
            Therefore the correct answre is E.<|STOP|>""",
            """-user:
            Question : A 66-year-old woman is seen in clinic for a new diagnosis of epilepsy; she has experienced recurrent partial seizures attributed to a left hemispheric stroke. Her medical history is most notable for atrial fibrillation (for which she takes warfarin), osteoporosis, and a history of kidney stones. Her primary care physician started her on phenytoin, which resulted in a rash. Which of the following is the most appropriate antiepileptic medication for this patient?
                A. carbamazepine
                B. levetiracetam
                C. primidone
                D. valproate
                E. zonisamide
            -assistant:
            Answer : - The question describes a 66-year-old woman with epilepsy secondary to a left hemispheric stroke, along with several comorbidities including atrial fibrillation, osteoporosis, and a history of kidney stones.
                - The patient had a hypersensitive reaction to phenytoin, making it unsuitable for further use.
                - Carbamazepine and primidone are potential options but pose challenges due to potential cross-reactivity with phenytoin and drug interactions with warfarin.
                - Zonisamide carries a risk of kidney stones, which is contraindicated in a patient with a history of calculi.
                - Valproate and enzyme-inducing medications are not ideal due to their association with bone density loss, exacerbating the patient's osteoporosis.
                - Among the choices, levetiracetam emerges as the most appropriate option due to its lower risk of hypersensitivity, minimal drug interactions with warfarin, and lack of exacerbation of osteoporosis or kidney stone formation.
            Therefore the final answer is B.<|STOP|>"""
        ],
        "q_form": "-user\nQuestion : {q}\n-assistant:\nAnswer : "
    }
}