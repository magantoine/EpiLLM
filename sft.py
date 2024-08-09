import os
import sys
import fire
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset

from typing import Tuple, Union
import torch
from trl import SFTTrainer

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

import huggingface_hub

ASSETS  = '/mloscratch/homes/vignoud/tuneinsight/assets'
MODELS = os.path.join(ASSETS, 'models/')
DATASETS = os.path.join(ASSETS, 'datasets/')


## PULL LOGIN
HF_TOKEN =  "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"
huggingface_hub.login(HF_TOKEN)


def main(
    model_name: str,
    output: str,
    dataset_name: str,
    val_set_size: Union[int, float] = 0.1,
    train_in_8bit: bool = False,
    max_input_length: int = 512,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    global_batch_size: int = 128,
    per_device_batch_size: int = 2,
    max_steps: int = -1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    save_total_limit: int = 15,
    eval_steps: int = 200,
    save_steps: int = 200,
    device_map: str = "auto",
    group_by_length: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "private_llm",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 100,
    flash_attention: bool = True,
    **kwargs
):
    output_dir = "scratch/home/magron"
    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")
    
    if use_wandb and len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Initial pad token:", tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token  #  '[PAD]' tokenizer.unk_token
    tokenizer.padding_side = "right"
    print("New pad token:", tokenizer.pad_token)
    # dataset_dir = os.path.join(DATASETS, dataset_name)
    # dataset = load_dataset(dataset_dir, num_proc=64)
    SYSTEM_PROMPT_QA = lambda is_mcq : f"You are a medical doctor taking the US Medical Licensing Examination. You need to demonstrate your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy. Show your ability to apply the knowledge essential for medical practice. {'For the following multiple-choice question, select one correct answer from A to E. Justify your answer and state the final answer letter.' if is_mcq else 'Answer the following question in the most clear and factual way and explain it.' } Base your answer on the current and standard practices referenced in medical guidelines."
    def prompt_template(q: str, stop_token:str=tokenizer.eos_token) -> str:
        is_mcq = "Question:" in q
        sys_prompt = SYSTEM_PROMPT_QA(is_mcq)
        return f"""-system:\n{sys_prompt}\n-user:{q.replace("Question:", "").strip()}\n-assistant:\n""".replace("###", stop_token)

    def load_sft_data(tok):
        """
            Prepare the raw dataset for, with tokenization and deterministic split
            in train and eval.

            args :    
                - tok (AutoTokenizer) : tokenizer
            
            returns :
                train and test dataset
        """
        dataset = load_dataset("cryptoni/epilepsy_guidelines_QA_v2", num_proc=64)
        dataset = dataset.map(lambda x : {
            # "question": ,
            "answer": prompt_template(x["question"]) + x["answer"] + tok.eos_token
        })

        # splits = ["train", "test"]
        # cols = ["questions", "answers"]
        max_length = min(tokenizer.model_max_length, 2048)
        print("Max_length", max_length)

        print(dataset)
        dataset = dataset.map(lambda x: {"token_ids": tok(x["answer"], max_length=max_length, truncation=True)["input_ids"]})
        print(dataset)
        dataset = dataset.map(lambda x: {"text":tok.decode(x["token_ids"], batched=True, num_proc=64)})
        print(dataset)
        # for split in splits: 
        #     dataset[split] = dataset[split].map(lambda x : tok(x["question"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length"), batched=True)
        #     dataset[split] = dataset[split].map(lambda x : {
        #         "labels" : tok(x["answer"], return_tensors="pt", max_length=min(tok.model_max_length, LLAMA2_MODEL_MAX_LENGTH), truncation=True, padding="max_length").input_ids
        #         }, batched=True).select_columns(["input_ids", "attention_mask", "labels"])
        
        return dataset.select_columns("text")
    dataset = load_sft_data(tokenizer)
    print(dataset)

    model_load_kwargs = {'device_map': device_map}
    if flash_attention:
        model_load_kwargs['attn_implementation'] = "flash_attention_2"

    if train_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=train_in_8bit, 
                                                llm_int8_has_fp16_weight=True,
                                                llm_int8_threshold=6)
        model_load_kwargs['torch_dtype'] = torch.bfloat16
        model_load_kwargs['quantization_config'] = quantization_config
    else:
        # loading the model with torch_dtype=torch.float16 with only fp16 and no LoRA leads
        # to `ValueError: Attempting to unscale FP16 gradients.`
        model_load_kwargs['torch_dtype'] = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"]   # because we added new tokens
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    gradient_accumulation_steps = global_batch_size // per_device_batch_size

    training_args = TrainingArguments(
        auto_find_batch_size=True,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="epoch",
        # evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="epoch",
        # eval_steps=eval_steps if val_set_size > 0 else None,
        # save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else "none",
        run_name=output if use_wandb else "none",
        save_safetensors=False,
        **kwargs
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_num_proc=64,
        max_seq_length=max_input_length,
        dataset_text_field='text',
        packing=True,
        eval_packing=True,
        tokenizer=tokenizer
    )

    if torch.__version__ >= "2" and sys.platform != "win32" and not use_lora:
        model = torch.compile(model)

    # finally, train
    trainer.train()

    if use_lora:
        model = model.merge_and_unload()

    # trainer.save_model(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

    ## PUsh LOGIN
    HF_TOKEN =  "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"
    huggingface_hub.login(HF_TOKEN)

    model_name = "llama-3-lora-15-epochs-best-checkpoint"
    tokenizer.push_to_hub(model_name)
    model.push_to_hub(model_name, safe_serialization=False)

if __name__ == "__main__":
    fire.Fire(main)
