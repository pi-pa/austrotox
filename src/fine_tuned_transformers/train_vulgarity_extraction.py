import argparse
import json
import os

import evaluate
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    RobertaTokenizerFast,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

language = "english"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
TASK_NAME = "vulgarity"

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
        "deepset-gbert-base": "deepset/gbert-base",
        "deepset-gbert-large": "deepset/gbert-large",
        "deepset-gelectra-base": "deepset/gelectra-base",
        "deepset-gelectra-large": "deepset/gelectra-large",
        "electra-large": "google/electra-large-discriminator",
        "roberta-base": "xlm-roberta-base", 
        "roberta-large": "FacebookAI/roberta-large"
    }

    if 'roberta' in MODELS_DICT[args.model_name]:
        tokenizer = RobertaTokenizerFast.from_pretrained(MODELS_DICT[args.model_name], add_prefix_space=True)
    else:  
        tokenizer = AutoTokenizer.from_pretrained(MODELS_DICT[args.model_name])  

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
        "warmup_steps": 200,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "evaluation_metric": "f1_binary",
        "patience": 3,

    }

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"vulgarity_labels"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label_to_id[label[word_idx]])
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    split = 0
    N_SPLITS = 10
    overall_results = {}
    for split in range(N_SPLITS):

        print('Model:', args.model_name)
        print('Split:', split)
        CONFIG[
            "output_dir"
        ] = f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}/split_{split}"
        CONFIG[
            "logging_dir"
        ] = f"{current_dir}/../../logs/{language}/{TASK_NAME}/{args.model_name}/split_{split}"
        CONFIG["split"] = split

        wandb.init(project=f"AustroTox-{TASK_NAME}", config=CONFIG, reinit=True)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if language == 'german':
            data_name = f"austro_tox_{split}"
        if language == 'english':
            data_name = f"english_tox_{split}"
        dataset = load_dataset(
            "austro_tox.py",
            name=data_name,
            data_file_path=f'{CONFIG["dataset_directory"]}split_{split}.json',
        )
        unique_labels = set(
            label for sample in dataset["train"]["vulgarity_labels"] for label in sample
        )
        label_list = list(unique_labels)
        print('Unique labels:', unique_labels)
        id_to_label = {i: name for i, name in enumerate(label_list)}
        label_to_id = {name: i for i, name in enumerate(label_list)}

        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]

        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
        test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

        model = AutoModelForTokenClassification.from_pretrained(
            CONFIG["model_name"],
            num_labels=len(label_list),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        model.to(device)

        seqeval = evaluate.load("seqeval")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            #results = seqeval.compute(
            #    predictions=true_predictions, references=true_labels
            #)

            lb = MultiLabelBinarizer()
            lb.fit(true_labels)
            lb.fit(true_predictions)
#
            true_labels_transformed = lb.transform(true_labels)
            true_predictions_transformed = lb.transform(true_predictions)

            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels_transformed, true_predictions_transformed, average="macro"
            )

            _, _, f1_all, _ = precision_recall_fscore_support(
                true_labels_transformed, true_predictions_transformed, average=None, zero_division=0
            )

            predicted_classes = list(lb.classes_)
            if 'VULGARITY' in predicted_classes:
                vul_i = list(lb.classes_).index('VULGARITY')
                f1_binary = f1_all[vul_i]
            else:
                f1_binary = 0

            #label_encoder = LabelEncoder()
            #label_encoder.fit(true_labels)
            #label_encoder.fit(true_predictions)
#
            #true_labels_encoded = label_encoder.transform(true_labels)
            #true_predictions_encoded = label_encoder.transform(true_predictions)
#
            #precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
            #    true_labels_encoded, true_predictions_encoded, pos_label=list(label_encoder.classes_).index('VULGARITY'), average="binary"
            #)

            return {
                "predicted_labels": true_predictions, 
                "true_labels": true_labels,
                #"precision_micro": results["overall_precision"],
                #"recall_micro": results["overall_recall"],
                #"f1_micro": results["overall_f1"],
                #"accuracy": results["overall_accuracy"],
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                #"precision_binary": precision_all[vul_i],
                #"recall_binary": recall_all[vul_i],
                "f1_binary": f1_binary,
                #"f1_group_all": f1_group
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
            tokenizer=tokenizer,
            data_collator=data_collator,
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
                #"test_precision": predictions.metrics["test_precision_binary"],
                #"test_recall": predictions.metrics["test_recall_binary"],
                "test_f1": predictions.metrics["test_f1_binary"],
                #"test_f1_vulgarity": predictions.metrics["test_f1_vulgarity"],
            }
        )
        wandb.finish()

        overall_results[split] = [results, predictions.metrics]

        print(predictions.metrics)
        with open(
            f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}/overall_results.json",
            "w",
        ) as fp:
            json.dump(overall_results, fp, indent=4)
