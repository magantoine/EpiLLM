from decorators import expose
import datasets
from datasets import Features, Value
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize


wtok = RegexpTokenizer(r'\w+')
FLAG = "epilep"


##################### PMC ###########################################################################
def advanced_filter(notes):
    """
        Takes clinical notes and advanced filter to try to see if
        the patient suffer from epilepsy.

        Also outputs the negative score from simple Naive Bayes (Vader Sentiment)
    """
    def check_negflag(negflag, sel_sents):
        return [negflag in [w.lower() for w in wtok.tokenize(sent.lower())] for sent in sel_sents]
    
    sents = sent_tokenize(notes)
    sel_sents = [
        sent for sent in sents if FLAG in sent.lower()
    ]

    

    return any(check_negflag("no", sel_sents)) or any(check_negflag("without", sel_sents))


def get_usable_PMC_patients():
    return datasets.load_dataset("zhengyun21/PMC-Patients", split="train").filter(
       lambda e : FLAG in e["patient"].lower() or FLAG in e["title"].lower()
    )

@expose
def get_pmc_patients():
    return datasets.load_from_disk("docs/pmc_patiens_fil.hf")





######################## PubMed ######################################################################

uround = lambda x : (x // 1_000) * 1_000
SPACE = " "* 50


def parse(split):
    split = list(split)
    start = split.pop(0)
    end = split.pop(-1)
    if(start != '[' or end != ']'):
        raise ValueError("Incorrect format")
    splits = "".join(split).split(":")
    if(len(splits) > 2):
        raise ValueError("Invalid number of splits")
    match splits :
        case ['']:
            return (0, 'all')
        case _:
            lb, ub = splits
            lb, ub = int(lb) if lb != '' else 0, int(ub) if ub != '' else 'all'
            if(ub != 'all' and lb > ub):
                raise ValueError("Incorrect bounds")
            return lb, ub
        

def get_iterator(lb, ub):
    match (lb, ub):
            case (0, 'all'):
                print("> fulliter <")
                iter = enumerate(datasets.load_dataset("pubmed", streaming=True)["train"])
            case (0, _):
                print(f"> iter up to {ub}<")
                iter = zip(range(ub), datasets.load_dataset("pubmed", streaming=True)["train"])
            case (_, 'all'):
                ### too slow : deprecated
                print(f"> iter from {lb}<")
                fulliter = enumerate(datasets.load_dataset("pubmed", streaming=True)["train"])
                i, _ = next(fulliter)
                while(i < lb):
                    i, _ = next(fulliter)
                iter = fulliter
            case (_, _):
                ### too slow : deprecated
                print(f"> iter from {lb} to {ub}<")
                fulliter = zip(range(ub), datasets.load_dataset("pubmed", streaming=True)["train"])
                i, _ = next(fulliter)
                while(i < lb):
                    i, _ = next(fulliter)
                iter = fulliter
    return iter





context_feat = Features({'title': Value(dtype='string', id=None), 'abstract': Value(dtype='string', id=None)})

@expose
def get_pubmed_ds(split, flag_filter=False, start_year=None):
    def pubmed_stream():
        iter = get_iterator(*parse(split))
        for pub in iter:
            first_pub_year = pub[1]["PubmedData"]["History"]["PubMedPubDate"]["Year"][0]
            if(start_year is None or first_pub_year >= start_year):
                print("Current year : ", first_pub_year, SPACE, end="\r")
                article = pub[1]["MedlineCitation"]["Article"]
                abstract = article["Abstract"]["AbstractText"]
                title = article["ArticleTitle"]
                filter = ((FLAG in title or FLAG in abstract) and abstract != '') if flag_filter else True
                if(filter):
                    yield {
                        "abstract" : abstract,
                        "title" : title
                    }
            else :
                print("Skipping : ", uround(pub[0]), SPACE, end="\r")
    
    return datasets.Dataset.from_generator(pubmed_stream, features=context_feat)


discharge_path = "docs/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv"
DISC_FLAGS = [
    "epilep"
]
radiology_path = "docs/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/radiology.csv"
RAD_FLAGS = [
    "brain",
    "cerebral"
]


@expose
def get_mimic_iv_notes(origin):
    disc = pd.read_csv(discharge_path)[["note_id", "text"]]
 
    match origin:
        case "disc":
            disc = pd.read_csv(discharge_path)[["note_id", "text"]]
            spec = [(disc, DISC_FLAGS)]
        case "rad":
            rad = pd.read_csv(radiology_path)[["note_id", "text"]]
            spec = [(rad, RAD_FLAGS)]
        case "all":
            disc = pd.read_csv(discharge_path)[["note_id", "text"]]
            rad = pd.read_csv(radiology_path)[["note_id", "text"]]
            spec = [(disc, DISC_FLAGS), (rad, RAD_FLAGS)]
        case _:
            raise ValueError("Unknown origin, it must be one of : ['disc', 'rad', 'all']")
    
    kept = []
    for df, flags in spec:
        kept.append(
            df[df.text.apply(lambda t : any(f in t for f in flags))]
        )
    return datasets.Dataset.from_pandas(
        pd.concat(kept)
    )
