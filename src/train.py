import argparse
import json
import os
from string import Template
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer

import delete_checkpoints
from delete_checkpoints import get_checkpoints
from prompts import LEOLMPrompts


# Define hyperparameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
fp16 = False
bf16 = False
per_device_train_batch_size = 8
per_device_eval_batch_size = 8
gradient_accumulation_steps = 8
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 1e-4 # 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
# save_steps = 25  # TODO
# eval_steps = save_steps
logging_steps = 2
max_seq_length = None
packing = False
device_map = {"": 0}


def load_dataset(path_dataset):
    with open(path_dataset, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset
    
    
def convert_to_prompt_format(dataset):
    """
    Create prompt format for dataset
    
    Args:
        dataset: list of dicts, each dict is a training example
            Keys: Index, Article_title, Comment, Label, Tags (subkeys: Tag, Token)
    Returns:
        prompt_format: list of dicts, each dict is a training example in prompt format
    """
    dataset_prompt_format = []
    labels = []
    for example in dataset:
        title = example["Article_title"]
        comment = example["Comment"]
        toxic = example["Label"]
        
        system_msg = LEOLMPrompts.get_multitask_system_msg()
        user_msg = LEOLMPrompts.get_user_msg(title=title, comment=comment)
        prompt = LEOLMPrompts.combine_system_and_user_msg(system_msg, user_msg)
        
        dataset_prompt_format.append({
            "index": example["Index"], 
            "prompt": prompt,
            "response": json.dumps({"Label": toxic, "Tags": example["Tags"]}, ensure_ascii=False)
        })
        labels.append(toxic)

    # count number of toxic and non-toxic examples
    num_toxic = sum(labels)
    num_non_toxic = len(labels) - num_toxic
    print(f"Number of toxic examples: {num_toxic}")
    print(f"Number of non-toxic examples: {num_non_toxic}")
    
    # dataset_prompt_format turn into a huggingface dataset
    columns = ["index", "prompt", "response"]
    data = {col: [example[col] for example in dataset_prompt_format] for col in columns}

    # Create a Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data)
    return hf_dataset


def train(train_dataset, validation_dataset, path_out_model, model_name, num_epochs):
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=path_out_model,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="all",
        evaluation_strategy="epoch",  # "epoch",
        save_strategy="epoch",  # "epoch"
        # eval_steps=eval_steps,
        # save_steps=save_steps,
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        dataset_text_field="prompt",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    trainer.train()
    trainer.model.save_pretrained(path_out_model)


def main(args):
    for n in range(args.num_cross_eval_splits):
        print("-" * 50)
        print(f"Training model on split {n}")
        dataset = load_dataset(os.path.join(args.path_splits_dir, f"split_{n}.json"))

        train_set = dataset["train"]
        print(f"Number of training examples: {len(train_set)}")
        validation_set = dataset["dev"]
        print(f"Number of validation examples: {len(validation_set)}")
        
        if args.limit_train_examples is not None:
            train_set = train_set[:args.limit_train_examples]
        if args.limit_dev_examples is not None:
            validation_set = validation_set[:args.limit_dev_examples]
        
        print("Convert training set to prompt format...")
        train_set_prompt_format = convert_to_prompt_format(train_set)
        print("Convert validation set to prompt format...")
        validation_set_prompt_format = convert_to_prompt_format(validation_set)
        
        output_dir = os.path.join(args.path_model_dir, f"split_{n}")
        os.makedirs(output_dir, exist_ok=True)
        train(train_set_prompt_format, validation_set_prompt_format, output_dir, args.hf_identifier, args.num_epochs)
        
        # delete non-best checkpoints
        del_args = argparse.Namespace()
        del_args.path_out_dir = output_dir
        if args.delete_non_best_checkpoints:
            delete_checkpoints.main(del_args)
        
        # Reload model to merge it
        checkpoints = get_checkpoints(output_dir)
        # best_checkpoint = delete_checkpoints.get_best_checkpoint(output_dir, checkpoints)
        # checkpoint_dir = os.path.join(output_dir, f"checkpoint-{best_checkpoint}")
        assert len(checkpoints) == 1
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-{checkpoints[0]}")
        
        print("Reload model and tokenizer...")
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir)
        model = model.merge_and_unload()

        # Reload tokenizer to save it
        tokenizer = AutoTokenizer.from_pretrained(args.hf_identifier, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Save the merged model
        print("Save merged model and tokenizer...")
        model.save_pretrained(f"{checkpoint_dir}_merged")
        tokenizer.save_pretrained(f"{checkpoint_dir}_merged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cross_eval_splits", type=int, help="Number of cross-evaluation splits")
    parser.add_argument("--path_splits_dir", type=str, help="Path to the directory containing the cross-evaluation splits")
    parser.add_argument("--path_model_dir", type=str, help="Path to the directory to save the trained models")
    parser.add_argument("--hf_identifier", type=str, help="HF identifier of the model to train")
    parser.add_argument("--delete_non_best_checkpoints", type=bool, default=True, help="Delete non-best checkpoints")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train the model")
    parser.add_argument("--limit_train_examples", type=int, default=None, help="Limit the number of training examples")
    parser.add_argument("--limit_dev_examples", type=int, default=None, help="Limit the number of dev examples")
    cmd_args = parser.parse_args()
    main(cmd_args)
