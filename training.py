import argparse
from pathlib import Path
import os
import math
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding)
from typing import (List, Tuple, Callable, Dict)
from datasets import (Dataset, concatenate_datasets)
from models import (get_pmc_patients, get_pubmed_ds, get_mimic_iv_notes)
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss

from dotenv import load_dotenv
load_dotenv()

if("DEVICE" in os.environ):
    DEVICE = torch.device(os.environ["DEVICE"])
    print("Running on device : ", DEVICE)
else :
    print("You need to set the device in the .env file. Resuming on CPU.")

DATAFUNCS = {
    "mimic": lambda : get_mimic_iv_notes("disc"),
    "pmc": get_pmc_patients,
    "pubmed": lambda : get_pubmed_ds(split="[:]", flag_filter=True)
}

INPUT_TYPES = {
    "lr" : float,
    "eps": float,
    "wrmp": float,
    "batch_size": int,
    "n_train_epoch": int,
    "verbose": bool
}

KNOWN_DATASETS = ["pubmed", "pmc", "mimic"]
VOCAB_SIZE = {
    "gpt2": 50257,
    "epfl-llm/meditron-7b": 32017
}

def tokenize_sample(sample: Dict[str, str],
                    tok: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
        Tokenize a sample with the tokenizer

        args :
            - sample (Dict[str, str]) : {'text' : 'The samples sentence'}
            - tok (AutoTokenize) : the used tokenizer
        
        returns :
            - {'input_ids': torch.tensor([...]), 'attention_mask': torch.tensor([...])}
    """
    ## context length : 2048 for meditron-7b, 512 for gpt2
    tokenized = tok(sample["text"], return_tensors="pt", max_length=2048, truncation=True, padding="max_length")
    tokenized["input_ids"] = tokenized["input_ids"].squeeze(0) ## sample by sample : squeeze
    tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0) ## sample by sample : squeeze
    return tokenized

def prepare_datasets(datasets: List[str],
                     tok: AutoTokenizer,
                     test_ratio: float=.2,
                     tokenize: bool=True) -> Tuple[Dataset, Dataset]:
    """
        Prepare the raw dataset, with tokenization and deterministic split
        in train and eval.

        args :
            - datasets (List[str]) : list of names of datasets to use
            - tok (AutoTokenizer) : tokenizer
            - test_ratio (float) : % to select to be eval dataset (deterministic)
            - tokenize (bool) : should it be tokenize, yes? no?
        
        returns :
            train and test dataset
    """
    datasets = [
        DATAFUNCS[dataset]()\
        .map(lambda s : tokenize_sample(s, tok) if tokenize else s)\
        .select_columns(["input_ids", "attention_mask"]) for dataset in datasets
    ] ## load and tokenize each dataset    
    splitted = [dataset.train_test_split(test_ratio) for dataset in datasets] ## split each of them
    train_datasets, test_datasets = zip(*[(d["train"], d["test"]) for d in splitted])
    return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)

def compute_loss(model: AutoModelForCausalLM,
                 inputs: Dict[str, str]) -> torch.Tensor:
    """
        Override of loss computation.

        args : 
            - model (AutoModelForCausalLM) :
            - inputs :

        returns :
            loss value, torch tensor with grad_fn
    """
    predictions = model(**inputs).logits ## model run and extract logits
    loss = CrossEntropyLoss()(predictions.float(),
                              one_hot(
                                  inputs["input_ids"],
                                  num_classes=VOCAB_SIZE["epfl-llm/meditron-7b"]
                            ).float()
                    ) ## Loss computation, comparing the logits and the one hot distrib
    return loss

def train_sft(**kwargs) -> None:
    raise NotImplementedError("SFT not yet implem")

def train_dpo(**kwargs) -> None:
    raise NotImplementedError("DPO not yet implem")

def train_cpt(datasets: List[str],
          save_dir: str,
          checkpoint: str,
          base_checkpoint: str,
          lr: float=2e-5,
          eps: float=1e-8,
          wrmp: float=.1,
          batch_size: int=4,
          n_train_epoch: int=1,
          verbose: bool=True) -> None:
          
    
    print("-", "Loading model and tokenize") if verbose else None
    llm = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16, # recommended on https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
        load_in_8bit=False,)
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    tok.add_special_tokens({'pad_token': '[PAD]'})

    data_collator = DataCollatorWithPadding(tok)
    
    print("-", "Loading dataset") if verbose else None
    train_dataset, test_dataset = prepare_datasets(datasets, tok, tokenize=True)
    train_dataset = Dataset.from_dict(train_dataset[:1])
    print("Training dataset : ") if verbose else None
    print(train_dataset) if verbose else None
    
    training_args = TrainingArguments(
                output_dir=Path(save_dir) / checkpoint,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=n_train_epoch, ## only 1 epoch
                evaluation_strategy="epoch",
                save_strategy="epoch",
                remove_unused_columns=False,
                fp16=True,
                save_steps=5_000,
                optim="adamw_torch_fused", # LLama 2 
                adam_beta1=0.9, # LLama 2 Meditron 7B
                adam_beta2=0.95, # LLama 2 Meditron 7B
                adam_epsilon=10e-5, # LLama 2 
                weight_decay=0.1, # Meditron 7B
                max_grad_norm=1.0, # Meditron 7B
                lr_scheduler_type="cosine", # Meditron 7B
                warmup_ratio=wrmp, # llama 2 (3%)
                learning_rate=3e-4, # Meditron 7B
        )

    trainer = Trainer(
        model=llm,
        train_dataset=train_dataset,
        tokenizer=tok,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset
    )
    trainer.compute_loss = compute_loss
    print("LAUNCH TRAINING")
    trainer.train()

    trainer.save_model()


def dispatch() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", help="Enter the names of the dataset to use uin : ['pubmed', 'pmc', 'mimic', 'all']", default="all", nargs="+")
    parser.add_argument("--save_dir", help="Directory in with you save the model checkpoints", default="checkpoints")
    parser.add_argument("--checkpoint", help="Name of the checkpoint folder", default="checkpoint")
    parser.add_argument("--base_checkpoint", help="Name of the base model", default="epfl-llm/meditron-7b")
    parser.add_argument("--lr", help="Learning rate", default=2e-5)
    parser.add_argument("--eps", help="Epsilon", default=1e-8)
    parser.add_argument("--wrmp", help="Warmup percentgage", default=.1)
    parser.add_argument("--batch_size", help="Batch size", default=4)
    parser.add_argument("--n_train_epoch", help="Number of train epoch", default=1)
    parser.add_argument("--verbose", help="Verbose", default=True)
    parser.add_argument("--type", help="Type of training in ['CPT', 'SFT', 'DPO']", default="CPT")
                       
    
    args = parser.parse_args()

    print("-", "Training with : ", ", ".join(args.datasets)) if args.verbose else None
    if(args.datasets == "all"):
        args.datasets = KNOWN_DATASETS
    if(any(d not in KNOWN_DATASETS for d in args.datasets)):
        raise ValueError(f"Dataset doesn't exist, must be one of : {KNOWN_DATASETS}")

    ## check the same dir
    save_dir = Path(args.save_dir)
    if(not save_dir.exists()):
        os.makedirs(save_dir)
        print("-", f"Created {save_dir} directory") if args.verbose else None
    
    print("-", "Will save at ", Path(save_dir) / args.checkpoint) if args.verbose else None

    if(not (save_dir / args.base_checkpoint).exists()):
        print("-", args.base_checkpoint, "is not a local checkpoint") if args.verbose else None
    
    print("-", "Will train from", args.base_checkpoint, "checkpoint") if args.verbose else None

    ## dispatch
    kwargs = vars(args)
    for input, cast in INPUT_TYPES.items():
        kwargs[input] = cast(kwargs[input])
    

    match args.type:
        case "DPO":
            del kwargs["type"]
            train_dpo(**kwargs)
        case "SFT":
            del kwargs["type"]
            train_sft(**kwargs)
        case "CPT":
            del kwargs["type"]
            train_cpt(**kwargs)
        case _ :
            raise ValueError("Unknown training type, must be one of DPO, SFT, CPT")
    

if __name__ == "__main__":
    dispatch()