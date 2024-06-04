import argparse
from pathlib import Path
import os
import torch
from typing import (List, Tuple, Dict)
from datasets import (Dataset, concatenate_datasets)
from models import (get_pmc_patients, get_pubmed_ds, get_mimic_iv_notes)
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding)
import time
from pynvml import *

from dotenv import load_dotenv
load_dotenv()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


## CONNECTING TO HUGGINGFACE API
import huggingface_hub
with open("/tmp/envfile", 'r') as f:
    HF_TOKEN = f.read().split("=")[1][1:-1]
print(HF_TOKEN)
huggingface_hub.login(HF_TOKEN)

### loading device
if("DEVICE" in os.environ):
    DEVICE = torch.device(os.environ["DEVICE"])
    print("Running on device : ", DEVICE)
else :
    print("You need to set the device in the .env file. Resuming on CPU.")


### functions to load datasets
SEED = 42
LLAMA2_MODEL_MAX_LENGTH = 4096
KNOWN_DATASETS = ["pubmed", "pmc", "mimic"]
DATAFUNCS = {
    "mimic": lambda : get_mimic_iv_notes("disc"),
    "pmc": get_pmc_patients,
    "pubmed": lambda : get_pubmed_ds(split="[:]", flag_filter=True)
}

### training arguments
TRAIN_CONFIG = {
    "adam_beta1": [0.9, "Adam Beta 1"],
    "adam_beta2": [0.95, "Adam Beta 2"],
    "adam_epsilon": [10e-5, "Adam Epsilon"],
    "max_grad_norm": [1.0, "Maximum norm for grad"],
    "lr_scheduler_type": ["cosine", "Type of scheduler for learning rate"],
    "wrmp": [0.3, "warmup ration"],
    "lr": [3e-4, "Learning rate"],
    "weight_decay": [0.1, "weight_decay"],
}


############################### DATA LOADING / PROCESSING #########################################################


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**3} GB.")


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
    tokenized = tok(
        sample["text"],
        return_tensors="pt",
        max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH),
        truncation=True,
        padding="max_length"
    )
    tokenized["input_ids"] = tokenized["input_ids"].squeeze(0) ## sample by sample : squeeze
    tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0) ## sample by sample : squeeze
    return tokenized


# def prepare_datasets(datasets: List[str],
#                      tok: AutoTokenizer,
#                      test_ratio: float=.2,
#                      tokenize: bool=True) -> Tuple[Dataset, Dataset]:
#     """
#         Prepare the raw dataset, with tokenization and deterministic split
#         in train and eval.

#         args :
#             - datasets (List[str]) : list of names of datasets to use
#             - tok (AutoTokenizer) : tokenizer
#             - test_ratio (float) : % to select to be eval dataset (deterministic)
#             - tokenize (bool) : should it be tokenize, yes? no?
        
#         returns :
#             train and test dataset
#     """
#     datasets = [
#         DATAFUNCS[dataset]()\
#         .map(lambda s : tokenize_sample(s, tok) if tokenize else s)\
#         .select_columns(["input_ids", "attention_mask"]) for dataset in datasets
#     ] ## load and tokenize each dataset    
#     splitted = [dataset.train_test_split(test_ratio, seed=SEED) for dataset in datasets] ## split each of them
#     train_datasets, test_datasets = zip(*[(d["train"], d["test"]) for d in splitted])
#     return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)

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
    #datasets = [
    #    DATAFUNCS[dataset]()\
    #    .map(lambda s : tokenize_sample(s, tok) if tokenize else s)\
    #    .select_columns(["input_ids", "attention_mask"]) for dataset in datasets
    #] ## load and tokenize each dataset    
    datasets = [
            DATAFUNCS[dataset]()\
            .map(lambda x : tok(x["text"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length") if tokenize else x, batched=True)\
            .select_columns(["input_ids", "attention_mask"])
    for dataset in datasets]
    splitted = [dataset.train_test_split(test_ratio, seed=SEED) for dataset in datasets] ## split each of them
    train_datasets, test_datasets = zip(*[(d["train"], d["test"]) for d in splitted])
    return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)


################################### TRAINING ############################################################################

def compute_loss(model: AutoModelForCausalLM,
                 inputs: Dict[str, str],
                 tokenizer: AutoTokenizer) -> torch.Tensor:
    """
        Override of loss computation.

        args : 
            - model (AutoModelForCausalLM) :
            - inputs :

        returns :
            loss value, torch tensor with grad_fn
    """
    VOCAB_SIZE = len(tokenizer)
    predictions = model(**inputs).logits ## model run and extract logits
    loss = CrossEntropyLoss()(predictions.float(),
                              one_hot(
                                  inputs["input_ids"],
                                  num_classes=VOCAB_SIZE
                            ).float()
                    ) ## Loss computation, comparing the logits and the one hot distrib
    return loss

def train_sft(**kwargs) -> None:
    """
        Supervised Fine Tuning (SFT) : used for simple alignement for tasks that
        are fairly easy to determine (MCQ, etc...)
    """
    raise NotImplementedError("SFT not yet implem")

def train_dpo(**kwargs) -> None:
    """
        Direct Preference Optimization (DPO) : used for alignement tasks that
        are easy to judge but hard to formalize (assistant, chat, surgery referral...)
    """
    raise NotImplementedError("DPO not yet implem")

def train_cpt(datasets: List[str],
          save_dir: str,
          checkpoint: str,
          base_checkpoint: str,
          lr: float,
          wrmp: float,
          adam_beta1: float,
          adam_beta2: float,
          adam_epsilon: float,
          weight_decay: float,
          max_grad_norm: float,
          lr_scheduler_type: str,
          n_train_epoch: int=1,
          batch_size: int=4,
          verbose: bool=True) -> None:
    """
        Continued Pre-Training (CPT) : we do next token prediction on decided datasets
        for domain adaptation and teach the model about the main concepts of epilepsy,
        diagnostics of patients, surgery, and research related topics.

        args :
            - save_dir (str) : save_dir
            - checkpoint (str) : checkpoint
            - base_checkpoint (str) : base_checkpoint
            - lr (float) : lr
            - wrmp (float) : wrmp
            - adam_beta1 (float) : adam_beta1
            - adam_beta2 (float) : adam_beta2
            - adam_epsilon (float) : adam_epsilon
            - weight_decay (float) : weight_decay
            - max_grad_norm (float) : max_grad_norm
            - lr_scheduler_type (str) : lr_scheduler_type
            - n_train_epoch (int) : n_train_epoch
            - batch_size (int) : batch_size
            - verbose (bool) : verbose

        return :
            - None
        
    """
          
    
    print("-", "Loading model and tokenize") if verbose else None
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    llm = AutoModelForCausalLM.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16, # recommended on https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
        load_in_8bit=False,
        # low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=False # set to true for inference
        # attn_implementation="flash_attention_2"
    )
    print_gpu_utilization()
    tok.pad_token = tok.eos_token
    data_collator = DataCollatorWithPadding(tok)
    
    print("-", "Loading dataset") if verbose else None
    train_dataset, test_dataset = prepare_datasets(datasets, tok, tokenize=True)
    #train_dataset = Dataset.from_dict(train_dataset[:2]) ## for subselection
    print("Training dataset : ") if verbose else None
    print(train_dataset) if verbose else None
    
    training_args = TrainingArguments(
                output_dir=Path(save_dir) / checkpoint,
                # per_device_train_batch_size=batch_size, # AUTO
                # per_device_eval_batch_size=batch_size,  # AUTO
                num_train_epochs=n_train_epoch, ## only 1 epoch
                evaluation_strategy="epoch",
                save_strategy="epoch",
                remove_unused_columns=False,
                save_steps=5_000,
                optim="adamw_torch_fused", # LLama 2 
                adam_beta1=adam_beta1, # LLama 2 Meditron 7B
                adam_beta2=adam_beta2, # LLama 2 Meditron 7B
                adam_epsilon=adam_epsilon, # LLama 2 
                weight_decay=weight_decay, # Meditron 7B
                max_grad_norm=max_grad_norm, # Meditron 7B
                lr_scheduler_type=lr_scheduler_type, # Meditron 7B
                warmup_ratio=wrmp, # llama 2 (3%)
                learning_rate=lr, # Meditron 7B
                ## TO ALLOW TRAINING ON A100
                bf16=True,
                tf32=True,
                torch_compile=True,
                auto_find_batch_size=True,
                debug="underflow_overflow",
                dataloader_num_workers=32,
                gradient_accumulation_steps=1, ## perhaps higher
                gradient_checkpointing=True

        )

    trainer = Trainer(
        model=llm,
        train_dataset=train_dataset,
        tokenizer=tok,
        args=training_args,
        data_collator=data_collator,
        eval_dataset=test_dataset
    )

    ## loss for next token prediction
    trainer.compute_loss = lambda model, inputs : compute_loss(model, inputs, tok)
    
    ## training launch
    print("LAUNCH TRAINING")
    trainer.train()
    
    ## save final model into the inputed save_dir/checkpoint
    trainer.save_model()



def check_datasets(datasets, verbose):
    print("-", "Training with : ", ", ".join(datasets)) if verbose else None
    if(datasets == "all"):
        datasets = KNOWN_DATASETS
    if(any(d not in KNOWN_DATASETS for d in datasets)):
        raise ValueError(f"Dataset doesn't exist, must be one of : {KNOWN_DATASETS}")
    
    return datasets

def check_save_dir(save_dir, verbose):
    save_dir = Path(save_dir)
    if(not save_dir.exists()):
        os.makedirs(save_dir)
        print("-", f"Created {save_dir} directory") if verbose else None

    return save_dir


def jupyter(**kwargs):
    ## leave container open
    time.sleep(1e8)

def dispatch() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", help="Enter the names of the dataset to use uin : ['pubmed', 'pmc', 'mimic', 'all']", default="all", nargs="+")
    parser.add_argument("--save_dir", help="Directory in with you save the model checkpoints", default="checkpoints")
    parser.add_argument("--checkpoint", help="Name of the checkpoint folder", default="checkpoint")
    parser.add_argument("--base_checkpoint", help="Name of the base model", default="epfl-llm/meditron-7b")
    parser.add_argument("--batch_size", help="Batch size", default=4, type=int)
    parser.add_argument("--verbose", help="Verbose", default=True, type=float)
    parser.add_argument("--n_train_epoch", help="Number of train epoch", default=1, type=int)
    parser.add_argument("--type", help="Type of training in ['CPT', 'SFT', 'DPO', 'jupyter']", default="CPT")

    for arg in TRAIN_CONFIG:
        parser.add_argument(f"--{arg}", help=TRAIN_CONFIG[arg][1], default=TRAIN_CONFIG[arg][0])                    
    
    args = parser.parse_args()

    ## check args
    args.datasets = check_datasets(args.datasets, args.verbose)
    save_dir = Path(check_save_dir(args.save_dir, args.verbose))

    print("-", "Will save at ", save_dir / args.checkpoint) if args.verbose else None
    if(not (save_dir / args.base_checkpoint).exists()):
        print("-", args.base_checkpoint, "is not a local checkpoint") if args.verbose else None
    print("-", "Will train from", args.base_checkpoint, "checkpoint") if args.verbose else None

    ## dispatch

    kwargs = vars(args)
    runtype = kwargs["type"]
    del kwargs["type"]
    

    match runtype:
        case "jupyter":
            jupyter(**kwargs)
        case "cmd":
            jupyter(**kwargs)
        case "DPO":
            train_dpo(**kwargs)
        case "SFT":
            train_sft(**kwargs)
        case "CPT":
            train_cpt(**kwargs)
        case _ :
            raise ValueError(
                "Unknown training type, must be one of DPO, SFT, CPT"
            )
    

if __name__ == "__main__":
    print("#"*200)
    print("Current user id : ", os.getuid())
    print("Current user name : ", os.popen('whoami').read())
    print("Current groups : ", os.popen('groups').read())
    print("#"*200)

    dispatch()