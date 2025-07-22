# Toxic Data

This repository accompanies the paper XYZ(link to paper).

## Clone the repository

```bash
$ git clone git@github.com:pi-pa/austrotox.git
$ cd toxic_data
```

## Create the virtual environment...

### ... using Python 3.9.2 (LLMs for now)

```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```


### ...using Conda (Transformers for now)

```bash
$ conda create -n env python=3.9 pip
$ conda activate env
$ pip install -r requirements-transformers.txt
```

## Execution Instructions to Reproduce the Experiments

Generate 10 splits for cross validation:
```bash
python3 src/generate_cross_eval_splits.py --num_splits 10 --path_data data/german/all_comments.json --path_splits_dir data/german/cross_eval_splits/ --path_output_dir data/german/train_dev_test
```

Optionally, validate the splits by running:
```bash
python3 src/check_split_overlap.py --num_versions 10 --path_data_dir data/german/train_dev_test
python3 src/compute_split_stats.py --num_splits 10 --path_data_dir data/german/train_dev_test
```

Fine-tune 10 models with a train/dev/test ratio of 8/1/1:
```bash
CUDA_VISIBLE_DEVICES=0 python3 src/train.py --num_cross_eval_splits 3 --path_splits_dir data/german/train_dev_test/ --path_model_dir data/german/models/ --hf_identifier LeoLM/leo-hessianai-7b-chat --num_epochs 3
```

Predict with the models on test splits:
```bash
python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_model_dir data/german/models/ --path_predictions_dir data/german/predictions --hf_identifier LeoLM/leo-hessianai-7b-chat --num_new_tokens 50 --gpu 1
```

Evaluate predictions and compute average scores:
```bash
python3 src/compute_metrics.py --path_predictions data/german/predictions/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/
```


# German

## Predict with ChatGPT 3.5
Predict German 0-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/german/train_dev_test/ --output_path data/german/predictions/gpt35/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-3.5-turbo-1106
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt35/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt35/0shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt35/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt35/0shot/ --multitask --consider_only_spans --span_requirement "both"
```

Predict 5-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/german/train_dev_test/ --output_path data/german/predictions/gpt35/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-3.5-turbo-1106
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt35/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt35/5shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt35/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt35/5shot/ --multitask --consider_only_spans --span_requirement "both"
```

## Predict with ChatGPT 4
Predict 0-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/german/train_dev_test/ --output_path data/german/predictions/gpt4/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-4-1106-preview
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt4/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt4/0shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt4/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt4/0shot/ --multitask --consider_only_spans --span_requirement "both"
```

Predict 5-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/german/train_dev_test/ --output_path data/german/predictions/gpt4/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-4-1106-preview
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt4/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt4/5shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/german/predictions/gpt4/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/gpt4/5shot/ --multitask --consider_only_spans --span_requirement "both"
```


## Mistral AI

### Use Generation
Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=7 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/mistral-instruct-v02-generation/0shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --num_new_tokens 5 --num_shots 0 
python3 src/compute_metrics.py --path_predictions data/german/predictions/mistral-instruct-v02-generation/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/mistral-instruct-v02-generation/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=6 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/mistral-instruct-v02-generation/5shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --num_new_tokens 5 --num_shots 5 
python3 src/compute_metrics.py --path_predictions data/german/predictions/mistral-instruct-v02-generation/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/mistral-instruct-v02-generation/5shot/
```

## Use Logits
Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=6,7 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/mistral-instruct-v02-logits/0shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --use_logits --num_shots 0
python3 src/compute_metrics.py --path_predictions data/german/predictions/mistral-instruct-v02-logits/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/mistral-instruct-v02-logits/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=4,5 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/mistral-instruct-v02-logits/5shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --use_logits --num_shots 5
python3 src/compute_metrics.py --path_predictions data/german/predictions/mistral-instruct-v02-logits/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/mistral-instruct-v02-logits/5shot/
```


## Predict with New Script and Prompt, Logit-Based
Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=6,7 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/leo-hessianai-7b-chat-logits/0shot/ --model_path_or_identifier LeoLM/leo-hessianai-7b-chat --use_logits --num_shots 0
python3 src/compute_metrics.py --path_predictions data/german/predictions/leo-hessianai-7b-chat-logits/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/leo-hessianai-7b-chat-logits/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=4,5 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/german/predictions/leo-hessianai-7b-chat-logits/5shot/ --model_path_or_identifier LeoLM/leo-hessianai-7b-chat --use_logits --num_shots 5
python3 src/compute_metrics.py --path_predictions data/german/predictions/leo-hessianai-7b-chat-logits/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/german/metrics/leo-hessianai-7b-chat-logits/5shot/
```

# English

## Predict with ChatGPT 3.5
Predict English 0-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt35/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-3.5-turbo-1106 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt35/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt35/0shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt35/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt35/0shot/ --multitask --consider_only_spans --span_requirement "both"
```

Predict 5-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt35/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-3.5-turbo-1106 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt35/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt35/5shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt35/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt35/5shot/ --multitask --consider_only_spans --span_requirement "both"
```

## Predict with ChatGPT 4
Predict 0-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt4/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-4-1106-preview --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt4/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt4/0shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt4/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt4/0shot/ --multitask --consider_only_spans --span_requirement "both"
```

Predict 5-shot:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt4/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-4-1106-preview --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt4/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt4/5shot/ --multitask
python3 src/compute_metrics.py --path_predictions data/predictions/english/gpt4/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/gpt4/5shot/ --multitask --consider_only_spans --span_requirement "both"
```

Chained ChatGPT calls for English data:
```bash
python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt35/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-3.5-turbo-1106 --language en; python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt35/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-3.5-turbo-1106 --language en; python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt4/0shot --multitask --num_shots 0 --random_seed 1 --model_name gpt-4-1106-preview --language en; python3 src/get_chatgpt_predictions.py --input_path data/english/train_dev_test/ --output_path data/predictions/english/gpt4/5shot --multitask --num_shots 5 --random_seed 1 --model_name gpt-4-1106-preview --language en
```


## Predict with Llama 3 Using Logits

## German
Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=5,6 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/predictions/german/Llama3_8B/0shot/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --use_logits --num_shots 0 --language de
python3 src/compute_metrics.py --path_predictions data/predictions/german/Llama3_8B/0shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/metrics/german/Llama3_8B/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=5,6 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/predictions/german/Llama3_8B/5shot/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --use_logits --num_shots 5 --language de
python3 src/compute_metrics.py --path_predictions data/predictions/german/Llama3_8B/5shot/ --path_true_labels data/german/train_dev_test/ --path_metrics data/metrics/german/meta-llama/Llama3_8B/5shot/
```

Predict 0-shot Multitask:
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/predictions/german/Llama3_8B/0shot-MT/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --num_shots 0 --language de --multitask
python3 src/compute_metrics.py --path_predictions data/predictions/german/Llama3_8B/0shot-MT/ --path_true_labels data/german/train_dev_test/ --path_metrics data/metrics/german/Llama3_8B/0shot-MT/
```

Predict 5-shot Multitask:
```bash
CUDA_VISIBLE_DEVICES=5,6 python3 src/predict.py --path_splits_dir data/german/train_dev_test/ --path_predictions_dir data/predictions/german/Llama3_8B/5shot-MT/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --use_logits --num_shots 5 --language de
python3 src/compute_metrics.py --path_predictions data/predictions/german/Llama3_8B/5shot-MT/ --path_true_labels data/german/train_dev_test/ --path_metrics data/metrics/german/meta-llama/Llama3_8B/5shot-MT/
```

## English
Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=2,4 python3 src/predict.py --path_splits_dir data/english/train_dev_test/ --path_predictions_dir data/predictions/english/Llama3_8B/0shot/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --use_logits --num_shots 0 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/Llama3_8B/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/Llama3_8B/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=4,5,6 python3 src/predict.py --path_splits_dir data/english/train_dev_test/ --path_predictions_dir data/predictions/english/Llama3_8B/5shot/ --model_path_or_identifier meta-llama/Meta-Llama-3-8B-Instruct --use_logits --num_shots 5 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/Llama3_8B/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/Llama3_8B/5shot/
```

## Predict with Mistral AI

Predict 0-shot:
```bash
CUDA_VISIBLE_DEVICES=4,5,6 python3 src/predict.py --path_splits_dir data/english/train_dev_test/ --path_predictions_dir data/predictions/english/mistral-instruct-v02-logits/0shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --use_logits --num_shots 0 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/mistral-instruct-v02-logits/0shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/mistral-instruct-v02-logits/0shot/
```

Predict 5-shot:
```bash
CUDA_VISIBLE_DEVICES=4,5,6 python3 src/predict.py --path_splits_dir data/english/train_dev_test/ --path_predictions_dir data/predictions/english/mistral-instruct-v02-logits/5shot/ --model_path_or_identifier mistralai/Mistral-7B-Instruct-v0.2 --use_logits --num_shots 5 --language en
python3 src/compute_metrics.py --path_predictions data/predictions/english/mistral-instruct-v02-logits/5shot/ --path_true_labels data/english/train_dev_test/ --path_metrics data/metrics/english/mistral-instruct-v02-logits/5shot/
```