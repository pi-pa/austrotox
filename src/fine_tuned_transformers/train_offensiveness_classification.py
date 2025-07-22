import argparse
import json
import os

import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    ElectraForSequenceClassification,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

language = "english"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
TASK_NAME = "offensiveness"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use",
        default="bert-base-german-cased",
    )

    args = parser.parse_args()

    MODELS_DICT = {
        "bert-base-german-cased": "bert-base-german-cased",
        "bert-base-german-dbmdz-cased": "dbmdz/bert-base-german-cased",
        "bert-base-uncased": "bert-base-uncased",
        "deberta-large": "microsoft/deberta-large",  # deberta-v3-large had memory errors
        "deberta-base": "microsoft/deberta-base",  
        "deepset-gbert-base": "deepset/gbert-base",
        "deepset-gbert-large": "deepset/gbert-large",
        "deepset-gelectra-base": "deepset/gelectra-base",
        "deepset-gelectra-large": "deepset/gelectra-large",
        "electra-large": "google/electra-large-discriminator",
        "roberta-base": "xlm-roberta-base", 
        "roberta-large": "FacebookAI/roberta-large",
        "twhin-bert-large": "Twitter/twhin-bert-large"
    }

    tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT[args.model_name])  

    def tokenize(batch):
        return tokenizer(
            batch["combined_text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"],
            return_tensors="pt",
        )

    current_dir = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(
        f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}", exist_ok=True
    )
    os.makedirs(
        f"{current_dir}/../../logs/{language}/{TASK_NAME}/{args.model_name}", exist_ok=True
    )

    CONFIG = {
        "dataset_directory": f"{current_dir}/../../data/{language}/train_dev_test/",
        "model_name": MODELS_DICT[args.model_name],
        "max_length": 512,
        "output_dir": "./results",
        "num_train_epochs": 10,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "warmup_steps": 400,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "evaluation_metric": "binary_f1",
        "patience": 3,
    }

    split = 0
    N_SPLITS = 10
    overall_results = {}
    for split in range(N_SPLITS):
        CONFIG[
            "output_dir"
        ] = f"{current_dir}/../../results/{TASK_NAME}/{args.model_name}/split_{split}"
        CONFIG[
            "logging_dir"
        ] = f"{current_dir}/../../logs/{TASK_NAME}/{args.model_name}/split_{split}"
        CONFIG["split"] = split

        wandb.init(project=f"AustroTox-{TASK_NAME}", config=CONFIG, reinit=True)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        dataset = load_dataset(
            "austro_tox.py",
            name=f"english_tox_{split}",  # austro_tox_
            data_file_path=f'{CONFIG["dataset_directory"]}split_{split}.json',
        )

        train_dataset = dataset["train"].map(tokenize, batched=True)
        val_dataset = dataset["validation"].map(tokenize, batched=True)
        test_dataset = dataset["test"].map(tokenize, batched=True)

        datasets = [train_dataset, val_dataset, test_dataset]
        for d in datasets:
            d.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        if 'roberta' in CONFIG["model_name"]:
            model = RobertaForSequenceClassification.from_pretrained(CONFIG["model_name"])
        if 'electra' in CONFIG["model_name"]:
            model = ElectraForSequenceClassification.from_pretrained(CONFIG["model_name"])
        else: 
            model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"])
        model.to(device)

        def compute_metrics(p):
            pred_labels = np.argmax(p.predictions, axis=1)
            true_labels = p.label_ids
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="binary", zero_division=0
            )
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels, pred_labels, average="macro", zero_division=0
            )
            return {
                "pred_labels": pred_labels.tolist(),
                "true_labels": p.label_ids.tolist(),
                "binary_precision": precision,
                "macro_precision": precision_macro,
                "binary_recall": recall,
                "macro_recall": recall_macro, 
                "binary_f1": f1,
                "macro_f1": f1_macro
            }

        training_args = TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["num_train_epochs"],
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
            warmup_steps=CONFIG["warmup_steps"],
            weight_decay=CONFIG["weight_decay"],
            logging_dir=CONFIG["logging_dir"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model=CONFIG["evaluation_metric"],
            greater_is_better=True,
            push_to_hub=False,
            report_to=["wandb"],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=CONFIG["patience"]
                )
            ],
        )

        trainer.train()

        results = trainer.evaluate()
        wandb.log(results)

        predictions = trainer.predict(test_dataset)
        wandb.log(
            {
                "test_precision": predictions.metrics["test_binary_precision"],
                "test_recall": predictions.metrics["test_binary_recall"],
                "test_f1": predictions.metrics["test_binary_f1"],
            }
        )
        wandb.finish()

        overall_results[split] = [results, predictions.metrics]

        print(overall_results)
        with open(
            f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}/overall_results.json",
            "w",
        ) as fp:
            json.dump(overall_results, fp, indent=4)
