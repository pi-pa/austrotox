import argparse
import json
import os
import random

import evaluate
import numpy as np
import torch
import transformers
import wandb
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import precision_recall_fscore_support

language = "english"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

TASK_NAME = "target_extraction"

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
        "roberta-large": "FacebookAI/roberta-large",
        "twhin-bert-large": "Twitter/twhin-bert-large"
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
        "learning_rate": 5e-5,  # tried 1e-4 for big models, resulted worse
        "warmup_steps": 200,
        "weight_decay": 0.01,  # tried 1e-5 for big models, resulted worse
        "logging_dir": "./logs",
        "evaluation_metric": "f1_micro",
        "patience": 5,  # was changed from 3
    }

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []
        for i, label in enumerate(examples[f"target_labels"]):
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            #previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                #elif (
                #    word_idx != previous_word_idx
                #):  # Only label the first token of a given word.
                #    label_ids.append(label_to_id[label[word_idx]])
                #else:
                #    label_ids.append(-100)
                else:
                    label_ids.append(label_to_id[label[word_idx]])
                #previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    #split = 0
    N_SPLITS = 10
    overall_results = {}
    for split in range(0, N_SPLITS):  # TODO: change this
        print('Model:', args.model_name)
        print('Split:', split)
        
        CONFIG[
            "output_dir"
        ] = f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}/split_{split}"
        CONFIG[
            "logging_dir"
        ] = f"{current_dir}/../../logs/{TASK_NAME}/{language}/{args.model_name}/split_{split}"
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
            label for sample in dataset["train"]["target_labels"] for label in sample
        )
        print('Unique labels:', unique_labels)
        label_list = list(unique_labels)  
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
            label2id=label_to_id
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

            results = seqeval.compute(
                predictions=true_predictions, references=true_labels
            )

            mlb = MultiLabelBinarizer()
            mlb.fit(true_labels)
            mlb.fit(true_predictions)

            true_labels_transformed = mlb.transform(true_labels)
            true_predictions_transformed = mlb.transform(true_predictions)

            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                true_labels_transformed, true_predictions_transformed, average="macro", zero_division=0
            )

            _, _, f1_all, _ = precision_recall_fscore_support(
                true_labels_transformed, true_predictions_transformed, average=None, zero_division=0
            )

            if len(f1_all) > 1:
                #i_f1_targets = list(range(4))
                #i_f1_targets.remove(list(mlb.classes_).index('O'))
                #f1_macro_t = np.mean([f1_all[i]] for i in i_f1_targets)
                f1_all_list = f1_all.tolist()
                f1_all_list.remove(max(f1_all_list))
                for _ in range(3 - len(f1_all_list)):
                    f1_all_list.append(0)
                f1_macro_t = np.mean(f1_all_list)
            else:
                f1_macro_t = 0

            #f1_targets = f1_score(
            #    true_labels_transformed, true_predictions_transformed, average="micro", labels=[
            #        "TARGET_GROUP", "TARGET_INDIVIDUAL", "TARGET_OTHER"
            #    ]
            #)
            #f1_targets = np.mean(f1_all[:3])
            
            return {
                "predicted_labels": true_predictions, 
                "true_labels": true_labels,
                #"precision_micro": results["overall_precision"],
                #"recall_micro": results["overall_recall"],
                "f1_micro": results["overall_f1"],
                #"accuracy": results["overall_accuracy"],
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
                "f1_targets": f1_macro_t,
                #"f1_targets": f1_targets
                "f1_all": f1_all.tolist()
            }

        training_args = TrainingArguments(
            output_dir=CONFIG["output_dir"],
            num_train_epochs=CONFIG["num_train_epochs"],
            per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
            per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
            warmup_steps=CONFIG["warmup_steps"],
            learning_rate=CONFIG["learning_rate"],
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

        from collections import Counter

        ## Count the occurrences of each label in the training set
        #label_counts = Counter([i for j in train_dataset['labels'] for i in j])
#
        #print("Label Counts:")
        #for label, count in label_counts.items():
        #    print(f"Label {label}: {count}")
#
        #samples_per_class = [label_counts[id_to_label[i]] for i in range(4)]
        #print("Samples per class:", samples_per_class)
#
        #class BalancedLossTrainer(Trainer):
        #    def compute_loss(self, model, inputs, return_outputs=False):
        #        labels = inputs.get("labels")
        #        outputs = model(**inputs)
        #        logits = outputs.get('logits')
        #        #loss_fct = MSELoss()
        #        #loss = loss_fct(logits.squeeze(), labels.squeeze())
        #        #return (loss, outputs) if return_outputs else loss
        #        # class-balanced binary cross-entropy loss
        #        bce_loss = Loss(
        #            loss_type="binary_cross_entropy",
        #            samples_per_class=samples_per_class,
        #            class_balanced=True
        #        )
        #        loss = bce_loss(logits.squeeze(), labels.squeeze())  #logits, labels
        #        return (loss, outputs) if return_outputs else loss
        
        trainer = Trainer(  # Trainer
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
                "test_precision": predictions.metrics["test_precision_macro"],
                "test_recall": predictions.metrics["test_recall_macro"],
                "test_f1": predictions.metrics["test_f1_macro"],
            }
        )
        wandb.finish()

        overall_results[split] = [results, predictions.metrics]

        print("test_precision", predictions.metrics["test_precision_macro"])
        print("test_recall", predictions.metrics["test_recall_macro"])
        print("test_f1", predictions.metrics["test_f1_macro"])
        print("test_f1_micro", predictions.metrics["test_f1_micro"])
        print("test_f1_all", predictions.metrics["test_f1_all"])
        
        with open(
            f"{current_dir}/../../results/{language}/{TASK_NAME}/{args.model_name}/overall_results.json",
            "w",
        ) as fp:
            json.dump(overall_results, fp, indent=4)
