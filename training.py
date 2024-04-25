import models
import argparse
from pathlib import Path
import os
import math
from dotenv import load_dotenv
load_dotenv()
import torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer,
                          TrainingArguments,
                          DataCollatorWithPadding,
                          AdamW,
                          get_linear_schedule_with_warmup)

from typing import List

if("DEVICE" in os.environ):
    DEVICE = torch.device(os.environ["DEVICE"])
    print("Running on device : ", DEVICE)
else :
    print("You need to set the device in the .env file. Resuming on CPU.")



def prepare_datasets(datasets, tok):
    pass



def train(datasets: List[str],
          save_dir: str,
          checkpoint: str,
          base_checkpoint: str,
          lr: float=2e-5,
          eps: float=1e-8,
          wrmp: float=.1,
          batch_size: int=4,
          n_train_epoch: int=1) -> None:
    

    llm = AutoModelForCausalLM.from_pretrained(base_checkpoint)
    tok = AutoTokenizer.from_pretrained(base_checkpoint)
    
    data_collator = DataCollatorWithPadding(tok)
    tokenized_datasets = prepare_datasets(datasets, tok)

    total_steps = 2 * math.ceil(len(tokenized_datasets["train"]) / batch_size)
    warmup_steps = int(wrmp * total_steps)

    optimizer = AdamW(llm.parameters(),
                  lr = lr, 
                  eps = eps
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=total_steps, num_warmup_steps=warmup_steps)
    

    training_args = TrainingArguments(
                output_dir=save_dir / checkpoint,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=n_train_epoch, ## only 1 epoch
                evaluation_strategy="epoch",
                save_strategy="epoch",
                remove_unused_columns=False,
            )

    trainer = Trainer(
        llm,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tok,
        optimizers=(optimizer, scheduler),        
    )



def dispatch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", help="Enter the names of the dataset to use uin : ['pubmed', 'pmc_patients', 'mimic', 'all']", default="all", nargs="+")
    parser.add_argument("--save_dir", help="Directory in with you save the model checkpoints", default="checkpoints")
    parser.add_argument("--checkpoint", help="Name of the checkpoint folder", default="checkpoint")
    parser.add_argument("--base_checkpoint", help="Name of the base model", default="epfl-llm/meditron-7b")
    parser.add_argument("--lr", help="Learning rate", default=2e-5)
    parser.add_argument("--eps", help="Epsilon", default=1e-8)
    parser.add_argument("--wrmp", help="Warmup percentgage", default=.1)
    parser.add_argument("--batch_size", help="Batch size", default=4)
    parser.add_argument("--n_train_epoch", help="Number of train epoch", default=1)
                       
    
    args = parser.parse_args()

    print("-", "Training with : ", ", ".join(args.datasets))

    ## check the same dir
    save_dir = Path(args.save_dir)
    if(not save_dir.exists()):
        os.makedirs(save_dir)
        print("-", f"Created {save_dir} directory")
    
    print("-", "Will save at ", Path(save_dir) / args.checkpoint)

    if(not (save_dir / args.base_checkpoint).exists()):
        print("-", args.base_checkpoint, "is not a local checkpoint")
    
    print("-", "Will train from", args.base_checkpoint, "checkpoint")

    ## dispatch
    train(**vars(args))
    

if __name__ == "__main__":
    dispatch()