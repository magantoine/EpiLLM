from decorators import expose
from typing import List, Any, Union, Dict

from .evaluate_utils import EmbSimModel
from torchmetrics.functional import pairwise_cosine_similarity
import evaluate as hf_evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer


wtok = RegexpTokenizer(r'\w+')
emb_sim_model = EmbSimModel("emilyalsentzer/Bio_ClinicalBERT")


@expose
def self_BLEU(sentences: List[str]) -> float:
    bleu = hf_evaluate.load("bleu")
    references = [sentences[:i] + sentences[i + 1:] for i in range(len(sentences))]
    return bleu.compute(predictions=sentences, references=references)


@expose
def embSim(sentences: List[str],
           format:str="matrix"
           ) -> Union[List[List[float]], List[float]]:
    if(format not in ["flat", "matrix"]):
        raise ValueError(f"Unsupported output format : {format}, must be in ['flat', 'matrix'].")
    pairwise = pairwise_cosine_similarity(emb_sim_model.get_emb(sentences)).detach().numpy()
    if(format == "matrix") :
        return pairwise
    elif(format == "flat"):
        t = ((np.ones(pairwise.shape) - np.tri(pairwise.shape[0], pairwise.shape[1])) * pairwise).flatten()
        t = t[t != 0]
        return t
    

@expose
def testBLEU(sentences: List[str],
             test_sentences: List[str]
             ) -> Dict[str, Any]:
    bleu = hf_evaluate.load("bleu")
    return bleu.compute(predictions=sentences, references=test_sentences)

@expose
def genlength(sentences: List[str]) -> List[int]:
    return np.array([len(wtok.tokenize(_)) for _ in sentences])



