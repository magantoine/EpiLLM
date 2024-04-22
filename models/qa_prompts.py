QA_PROMPTS = {
    "direct_answer_align" : {
        "system": "You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. For the following multiple-choice question, select one correct answer from A to E. Base your answer on the current and standard practices referenced in medical guidelines.",
        "shots": [ ## 3 samples
            """-user
            Question : Which of the following is the major role of the neurologist after a diagnosis of psychogenic nonepileptic events?,
                    A. appropriately weaning antiepileptic drugs
                    B. describing events in ambiguous terms
                    C. ensuring that patients continue to refrain from driving a motor vehicle
                    D. ordering follow-up EEG in 6 to 12 months
                    E. prescribing a selective serotonin reuptake inhibitor or other antidepressant
            -assistant:
            Answer : A<|STOP|>""",
            """-user:
            Question : Deep brain stimulation for drug-resistant epilepsy targets which of the following structures?
                    A. anterior nucleus of the thalamus
                    B. centromedian nucleus of the thalamus
                    C. cingulate gyrus
                    D. hippocampus
                    E. internal segment of the globus pallidus
            -assistant:
            Answer : A<|STOP|>""",
            """-user:
            Question : According to the International League Against Epilepsy\’s 2010 revision of its seizure classification system, which of the following factors is most helpful in initially focusing the scope of diagnostic possibilities for specific electroclinical syndromes?
                    A. age at onset
                    B. family seizure history
                    C. frequency of spells
                    D. history of toxic exposure
                    E. severity of the disorder
            -assistant:
            Answer : A<|STOP|>"""
        ],
        "q_form": "-user\nQuestion : {q}\n-assistant:\nAnswer : "
    },
    "direct_answer_align":{
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