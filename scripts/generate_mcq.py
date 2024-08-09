import sys
sys.path.append("..")
from models import OpenAIGPT
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle



gpt = OpenAIGPT("gpt-4", 1)
def create_mcq(q, a, letter):
    return gpt.query([
        # {
        #     "role": "system",
        #     "content": "You're a neurologist preparing an exam for the best students in the country. Transform this question into an MCQ question. You will write a creative and complicated MCQ and finish by writing the answer"
        # },
        {
            "role": "system",
            "content": f"""You're a neurologist preparing an exam for the best students in the country. Transform this question into an MCQ question.
            You will write a creative and complicated MCQ and finish by writing the answer. The correct answer should be {letter} and should formatted exactly as follows :
            
            Question:
             insert the question here
            Answer:
                Justify the answer here developing your thoughts.
                Therefore the correct answer is {letter}"""
        },
        {   
            "role": "user",
            "content": 
            f""" 
                Question: {q}
                Answer: {a}
            """
        }
    ]
    )


import nltk
nltk.download("punkt")
FLAGS = ["Therefore", "correct", "answer"]
def match_score(sentence):
    return len([flag for flag in FLAGS if flag in sentence]) / len(FLAGS)

def simple_extract(ans_sentence):
    sel = [l for l in ["A", "B", "C", "D", "E"] if l in ans_sentence]
    return sel[0] if sel != [] else "-1"

def csa2(pred):
    sent_text = nltk.sent_tokenize(pred.replace("\n", "??"))
    sentence_score = sorted([[sent, match_score(sent)] for sent in sent_text], key=lambda _: _[1], reverse=True)
    if(len(sentence_score) == 0):
        return "-1"
    sentence, score = sentence_score[0]
    if(score == 0):
        return "-1"
    return simple_extract(sentence)



mcq_sample = pd.read_csv("generated_mcq.csv")

mcqs = []
for i, (q, a, letter, mcq) in tqdm(list(enumerate(mcq_sample[["q", "a", "letter", "mcq"]].values))):
    # print(mcq)
    if(csa2(mcq) == "-1"):
        if(i % 100 == 0):
            ## 100 checkpoint
            with open("generated.pkl", "wb") as f:
                pickle.dump(mcqs, f)
        # print("*"*150)
        # print("NEW MCQ")
        # print(new_mcq := create_mcq(q, a, letter))

        # mcqs.append(new_mcq)
    # else :
    #     print("*"*150)
    #     print("KEEPING")
    #     print(mcq)
    #     print("Well formatted, skipping")
    #     mcqs.append(mcq)
    
    


mcq_sample["mcq2"] = mcqs


mcq_sample.to_csv("generated_mcq.csv")