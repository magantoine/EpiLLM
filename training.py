import argparse
from pathlib import Path
import os
import torch
from typing import (List, Tuple, Dict)
from datasets import (Dataset, concatenate_datasets, load_dataset)
from models import (get_pmc_patients, get_pubmed_ds, get_mimic_iv_notes)
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          LlamaForCausalLM)
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from trl import SFTTrainer
import time
from pynvml import *

from dotenv import load_dotenv
load_dotenv()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


## CONNECTING TO HUGGINGFACE API
# import huggingface_hub
# with open("/tmp/envfile", 'r') as f:
#     HF_TOKEN = f.read().split("=")[1][1:-1]
# print(HF_TOKEN)
# huggingface_hub.login(HF_TOKEN)

### loading device
if("DEVICE" in os.environ):
    DEVICE = torch.device(os.environ["DEVICE"])
    print("Running on device : ", DEVICE)
else :
    print("You need to set the device in the .env file. Resuming on CPU.")


### functions to load datasets
SEED = 42
LLAMA2_MODEL_MAX_LENGTH = 2048


def get_pubmed():
    epi_pubmed = load_dataset("cryptoni/epi_pubmed")
    train, test = epi_pubmed["train"], epi_pubmed["test"]
    train = train.rename_column("abstract", "text")
    test = test.rename_column("abstract", "text")
    return train.select_columns("text"), test.select_columns("text")

KNOWN_DATASETS = ["pubmed", "pmc", "mimic"]
DATAFUNCS = {
    "mimic": lambda : get_mimic_iv_notes("disc"),
    "pmc": get_pmc_patients,
    "pubmed": get_pubmed
}

### training arguments
TRAIN_CONFIG = {
    "adam_beta1": [0.9, "Adam Beta 1"],
    "adam_beta2": [0.95, "Adam Beta 2"],
    "adam_epsilon": [10e-5, "Adam Epsilon"],
    "max_grad_norm": [1.0, "Maximum norm for grad"],
    "lr_scheduler_type": ["cosine", "Type of scheduler for learning rate"],
    "wrmp": [0.3, "warmup ration"],
    "lr": [5e-6, "Learning rate"],
    "weight_decay": [0.1, "weight_decay"],
}


SFT_CONFIG = {
    "lora": [True, "Using LoRA ?"],
    "lora_r": [8, "LoRA r"],
    "lora_alpha": [16, "LoRA Alpha"],
    "lora_dropout": [0.1, "LoRA Dropout"],
    "lora_target_modules": [["q_proj", "v_proj"], "LoRA Target Modules"],
    "sft_learning_rate": [2e-5, "Learning for SFT"],
    "sft_optim": ["adamw_torch", "optimizer for SFT"],
    "sft_lr_scheduler_type": ["cosine", "Scheduler for lr for SFT"],
    "sft_wrmp": [0.05, "warmup for sft"],
    "sft_adam_beta1": [0.9, "Adam Beta 1"],
    "sft_adam_beta2": [0.95, "Adam Beta 2"],
    "sft_adam_epsilon": [10e-5, "Adam Epsilon"],
    "sft_max_grad_norm": [1.0, "Maximum norm for grad"],
    "sft_packing": [True, "SFT Packing"],
}


############################### DATA LOADING / PROCESSING #########################################################


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**3} GB.")



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
            .map(lambda x : tok(x["text"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length") if tokenize else x, batched=True)\
            .select_columns(["input_ids", "attention_mask"])
    for dataset in datasets]
    splitted = [dataset.train_test_split(test_ratio, seed=SEED) for dataset in datasets] ## split each of them
    train_datasets, test_datasets = zip(*[(d["train"], d["test"]) for d in splitted])
    return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)




def load_sft_data(tok):
    """
        Prepare the raw dataset for, with tokenization and deterministic split
        in train and eval.

        args :    
            - tok (AutoTokenizer) : tokenizer
        
        returns :
            train and test dataset
    """
    dataset = load_dataset("cryptoni/epilepsy_guidelines_QA")

    splits = ["train", "test"]
    cols = ["questions", "answers"]
    for split in splits: 
        dataset[split] = dataset[split].map(lambda x : tok(x["questions"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length"), batched=True)
        dataset[split] = dataset[split].map(lambda x : {
            "labels" : tok(x["answers"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length").input_ids
            }, batched=True).select_columns(["input_ids", "attention_mask", "labels"])
    
    return dataset["train"], dataset["test"]

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


def train_dpo(**kwargs) -> None:
    """
        Direct Preference Optimization (DPO) : used for alignement tasks that
        are easy to judge but hard to formalize (assistant, chat, surgery referral...)
    """
    raise NotImplementedError("DPO not yet implem")


def train_sft(base_checkpoint: str,
               lora: bool,
               save_dir: str,
               checkpoint: str,
               sft_adam_beta1: float ,
               sft_adam_beta2: float ,
               sft_adam_epsilon: float,
               sft_max_grad_norm: float,
               sft_lr_scheduler_type: float,
               sft_optim: str,
               sft_wrmp: float,
               lora_alpha: float,
               lora_target_modules: List[str],
               lora_dropout: float,
               lora_r:int,
               n_train_epoch: int=1,
               sft_lr: float=5e-6,
               verbose: bool=True,
               sft_packing: bool=True,
               **kwargs) -> None:
    """
        Supervised Fine-Tunin with Low-Rank Adaptation (SFT-LoRA) : used for alignement tasks that
        are easy to judge but hard to formalize (assistant, chat, surgery referral...)
    """

    print("-", "Loading model and tokenize") if verbose else None
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    llm = LlamaForCausalLM.from_pretrained(
        base_checkpoint,
        torch_dtype=torch.bfloat16, # recommended on https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
        load_in_8bit=False,
        # low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=False, # set to true for inference
        attn_implementation="flash_attention_2"
    )
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    data_train, data_test = load_sft_data(tok)
    print(data_train)
    print_gpu_utilization()

    if lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"]   # because we added new tokens
        )
        llm.enable_input_require_grads()
        llm = get_peft_model(llm, lora_config)
        llm.print_trainable_parameters()
        llm = prepare_model_for_kbit_training(llm)
        llm = get_peft_model(llm, lora_config)
    
    
    training_args = TrainingArguments(
                output_dir=Path(save_dir) / checkpoint,
                num_train_epochs=n_train_epoch, ## only 1 epoch
                evaluation_strategy="no",
                save_strategy="no",
                remove_unused_columns=False,
                save_steps=5_000,
                optim=sft_optim, # LLama 2 
                adam_beta1=sft_adam_beta1, # LLama 2 Meditron 7B
                adam_beta2=sft_adam_beta2, # LLama 2 Meditron 7B
                adam_epsilon=sft_adam_epsilon, # LLama 2 
                max_grad_norm=sft_max_grad_norm, # Meditron 7B
                lr_scheduler_type=sft_lr_scheduler_type, # Meditron 7B
                warmup_ratio=sft_wrmp, # llama 2 (3%)
                learning_rate=sft_lr, # Meditron 7B
                ## TO ALLOW TRAINING ON A100
                bf16=True,
                tf32=True,
                torch_compile=True,
                auto_find_batch_size=True,
                debug="underflow_overflow",
                dataloader_num_workers=16,
                gradient_accumulation_steps=32, ## perhaps higher
                gradient_checkpointing=True

        )

    
    trainer = SFTTrainer(
        llm,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_test,
        dataset_num_proc=64,
        max_seq_length=LLAMA2_MODEL_MAX_LENGTH,
        # dataset_text_field='input_ids',
        # formatting_func=formatting_func,
        packing=sft_packing,
        eval_packing=sft_packing,
        tokenizer=tok
    )

    ## training launch
    print("LAUNCH TRAINING")
    trainer.train()

    if lora:
        llm = llm.merge_and_unload()
    
    ## save final model into the inputed save_dir/checkpoint
    trainer.save_model()


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
        use_cache=False # set to true for inference
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
                evaluation_strategy="no",
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
                dataloader_num_workers=16,
                gradient_accumulation_steps=32, ## perhaps higher
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
    # trainer.compute_loss = lambda model, inputs : compute_loss(model, inputs, tok)
    
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
    
    for arg in SFT_CONFIG:
        parser.add_argument(f"--{arg}", help=SFT_CONFIG[arg][1], default=SFT_CONFIG[arg][0])                    
    
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