import models
import argparse
from pathlib import Path
import os
import math
from dotenv import load_dotenv
load_dotenv()
import torch
from peft import LoraConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          Seq2SeqTrainer,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          AdamW,
                          get_linear_schedule_with_warmup)

from typing import (List,
                    Tuple,
                    Callable)
from datasets import (Dataset,
                      concatenate_datasets)
from models import (get_pmc_patients,
                    get_pubmed_ds,
                    get_mimic_iv_notes)

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


def tokenize_sample(sample, tok):
    # return tok(sample["text"], return_tensors="pt", max_length=512, truncation='max_length', padding=True)
    
    toked = tok(sample["text"], max_length=512, truncation='longest_first', padding=True)
    toked["text"] = sample["text"]
    
    return toked

def prepare_datasets(datasets: List[str],
                     tok: AutoTokenizer,
                     test_ratio: float=.2,
                     tokenize: bool=True) -> Tuple[Dataset, Dataset]:
    datasets = [DATAFUNCS[dataset]() for dataset in datasets]
    [dataset.set_transform(lambda s : tokenize_sample(s, tok)) for dataset in datasets] if tokenize else None
    splitted = [dataset.train_test_split(test_ratio) for dataset in datasets]
    train_datasets, test_datasets = zip(*[(d["train"], d["test"]) for d in splitted])
    return concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)



def train(datasets: List[str],
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
    llm = AutoModelForCausalLM.from_pretrained(base_checkpoint)
    tok = AutoTokenizer.from_pretrained(base_checkpoint)

    data_collator = DataCollatorWithPadding(tok)
    
    print("-", "Loading dataset") if verbose else None
    train_dataset, test_dataset = prepare_datasets(datasets, tok, tokenize=True)
    print("Training dataset : ") if verbose else None
    print(train_dataset) if verbose else None
    print(train_dataset[0]) if verbose else None
    
    total_steps = 2 * math.ceil(len(train_dataset) / batch_size)
    warmup_steps = int(wrmp * total_steps)

    optimizer = AdamW(llm.parameters(),
                  lr = lr, 
                  eps = eps
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
    

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
            )

    # trainer = Seq2SeqTrainer(
    #     llm,
    #     training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     data_collator=data_collator,
    #     tokenizer=tok,
    #     optimizers=(optimizer, scheduler),        
    # )

    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=llm,
        train_dataset=train_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tok,
        args=training_args,
        packing=False,
    )
    print("LAUNCH TRAINING")
    trainer.train()

def dispatch():
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
                       
    
    
    args = parser.parse_args()

    print("-", "Training with : ", ", ".join(args.datasets)) if args.verbose else None
    if(args.datasets == "all"):
        args.datasets = ["pubmed", "pmc", "mimic"]

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
    train(**kwargs)
    

if __name__ == "__main__":
    dispatch()