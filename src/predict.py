import argparse
import json
import os
import re
import torch
import random
from typing import Dict, List
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase, AutoConfig

import prompts


use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False


def get_shots(data: List[Dict], n: int, multitask: bool, language: str) -> Dict[str, str]:
    """Sample n random instances from the given data."""
    samples = random.sample(data, n)
    sample_dict = {}
    for i in range(n):
        if language == "de":
            sample_dict[f"title_{i+1}"] = f"Title: {samples[i]['Article_title']}"
        sample_dict[f"comment_{i+1}"] = f"Comment: {samples[i]['Comment']}"
        if multitask:
            sample_dict[f"response_{i+1}"] = json.dumps({"Label": samples[i]["Label"], "Tags": samples[i]['Tags']}, ensure_ascii=False)
        else:
            sample_dict[f"response_{i+1}"] = json.dumps(int(samples[i]["Label"]), ensure_ascii=False)
    return sample_dict


def generate_prediction_multitask(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, title: str, comment: str, shots: Dict[str, str], device: str, use_logits: bool, num_new_tokens: int, language: str) -> str:
    # Identify the model identifier
    model_id = model.config._name_or_path

    # Construct the input prompt according to the model type and the shot learning scenario
    if "mistralai/Mistral-7B-Instruct-v0.2" in model_id or "Llama-3" in model_id:

        # Handling for models expecting a structured input (e.g., a series of user-system exchanges)
        prompt = prompts.ChatGPTPrompts.get_multitask_system_msg(num_shots=len(shots)/5, shots=shots, language=language)
        user_prompt = prompts.ChatGPTPrompts.get_user_msg(title=title, comment=comment, num_shots=len(shots)/5, language=language)
        prompt = prompt + user_prompt
    elif "LeoLM" in model_id:

        # Handling for models expecting a more open-ended input
        system_msg = prompts.LEOLMPrompts.get_multitask_system_msg(num_shots=len(shots)/5, shots=shots, language=language)
        user_msg = prompts.LEOLMPrompts.get_user_msg(title=title, comment=comment, num_shots=len(shots)/5, language=language)
        prompt = prompts.LEOLMPrompts.combine_system_and_user_msg(system_msg, user_msg)
    else:
        raise ValueError(f"Unexpected model id: {model_id}")

    # Encode the prompt
    encoded_input = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    if use_logits:

        # If using logits for prediction, generate a response by looking at logits of possible outputs
        outputs = model(**encoded_input)
        logits = outputs.logits
        
        # Logic to extract and interpret logits would be similar to what's in generate_prediction
        # This is a placeholder for the logic that would be necessary
        # Handle interpretation of logits here...
        raise NotImplementedError("Logit-based interpretation for multitask is not implemented.")
    else:
        # Generate tokens for the output response
        generated_ids = model.generate(**encoded_input, max_new_tokens=num_new_tokens, pad_token_id=tokenizer.pad_token_id)
        decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract and return the structured response from the model's generated text
        try:
            response_json = json.loads(decoded_output)
            return response_json
        except json.JSONDecodeError:
            # Handling cases where decoding fails
            raise ValueError("Failed to decode the multitask response from the model.")

    return decoded_output


def generate_prediction(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, title: str, comment: str, shots: Dict[str, str], device: str, use_logits: bool, num_new_tokens: int, language: str) -> str:

    # get hf identifier of the model
    model_id = model.config._name_or_path

    # apply chat template and tokenizer depending on the model
    if "mistralai/Mistral-7B-Instruct-v0.2" in model_id or "Llama-3" in model_id:

        # Constructing the conversation history with shots
        if shots == {}:
            if language == "de":
                content_to_classify = prompts.Prompts.user_msg_template_0shot_german.format(title=title, comment=comment)
            elif language == "en":
                content_to_classify = prompts.Prompts.user_msg_template_0shot_english.format(comment=comment)
        else:
            if language == "de":
                content_to_classify = prompts.Prompts.user_msg_template_5shot_german.format(title=title, comment=comment)
            elif language == "en":
                content_to_classify = prompts.Prompts.user_msg_template_5shot_english.format(comment=comment)
        if len(shots) == 0:
            if language == "de":
                messages = [{"role": "user", "content": prompts.Prompts.system_msg_toxicity_german + content_to_classify}]
            elif language == "en":
                messages = [{"role": "user", "content": prompts.Prompts.system_msg_toxicity_english + content_to_classify}]
        else:
            if language == "de":
                messages = [{"role": "user", "content": prompts.Prompts.system_msg_toxicity_german + prompts.Prompts.examples_german.format(**shots) + content_to_classify}]
            elif language == "en":
                messages = [{"role": "user", "content": prompts.Prompts.system_msg_toxicity_english + prompts.Prompts.examples_english.format(**shots) + content_to_classify}]
        tokenizer.add_special_tokens({"pad_token": "<|eot_id|>"})
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True, padding=True, truncation=True, add_special_tokens=True)

    
    elif "LeoLM" in model_id:
        system_msg = prompts.LEOLMPrompts.get_toxicity_system_msg(num_shots=len(shots) / 3, shots=shots)
        user_msg = prompts.LEOLMPrompts.get_user_msg(title=title, comment=comment, num_shots=len(shots) / 3)
        prompt = prompts.LEOLMPrompts.combine_system_and_user_msg(system_msg, user_msg)
        encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True)["input_ids"]
    else:
        raise ValueError(f"Unexpected model id: {model_id}")
    
    if use_logits:
        model_inputs = encodeds.to(device)
        input_ids = model_inputs
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_token_id).type(input_ids.dtype)
        attention_mask = attention_mask.to(device)

        # Get model output
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False, return_dict=True)
        logits = outputs.logits

        # Extract logits for the last non-padded token
        # We are interested in the token before the padding begins
        mask = attention_mask.bool()
        last_token_logits = logits[mask, :].squeeze()[-1, :]

        # Get probabilities from logits
        probs = torch.softmax(last_token_logits, dim=-1)

        # Token ids for "0" and "1"
        token_id_0 = tokenizer.convert_tokens_to_ids("0")
        token_id_1 = tokenizer.convert_tokens_to_ids("1")

        # Compare probabilities
        prob_0 = probs[token_id_0].item()
        prob_1 = probs[token_id_1].item()
        prediction = 0 if prob_0 > prob_1 else 1
        return prediction
    else:
        model_inputs = encodeds.to(device)
        input_ids = model_inputs
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        attention_mask = (input_ids != pad_token_id).type(input_ids.dtype)
        attention_mask = attention_mask.to(device)
        generated_ids = model.generate(model_inputs, attention_mask=attention_mask, max_new_tokens=num_new_tokens, do_sample=False, pad_token_id=2)
        decoded = tokenizer.batch_decode(generated_ids)
        if model_id == "mistralai/Mistral-7B-Instruct-v0.2" or "Llama-3" in model_id:
            try:
                response = int(decoded[0].split("[/INST] ")[1])
                return response
            except:
                response = decoded[0].split("[/INST]")[1]

                # use regex to extract the number
                numbers = re.findall(r"\d", response)
                if len(numbers) != 1:
                    raise ValueError(f"Unexpected number of numbers in decoded response: {response}")
                assert numbers[0] in ["0", "1"]
                return int(numbers[0])
        elif "LeoLM" in model_id:
            try:
                response = int(decoded[0].split("\n<|im_start|>assistant\n")[1])
                return response
            except:
                response = decoded[0].split("\n<|im_start|> assistant\n")[1]
                # use regex to extract the number
                numbers = re.findall(r"\d", response)
                if len(numbers) != 1 or numbers[0] not in ["0", "1"]:
                    print(response)
                    return None
                return int(numbers[0])


def main(args):
    # Load and possibly customize the configuration
    config = AutoConfig.from_pretrained(args.model_path_or_identifier)

    # Configure BnB quantization
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    # Load model with BnB quantization and device_map="auto"
    print(f"Load model from HF hub or cache: {args.model_path_or_identifier}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path_or_identifier,
        quantization_config=bnb_config,
        config=config,
        device_map="auto"  # Automatically distribute the model
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_identifier)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    for fname in os.listdir(args.path_splits_dir):
        split = fname.replace(".json", "")
        # if int(split[-1]) < 7:
        #     continue
        print(20*"=" + f" Generating testset predictions for {split} " + 20*"=")

        file_cur_split = os.path.join(args.path_splits_dir, f"{split}.json")
        with open(file_cur_split, "r") as f:
            data_cur_split = json.load(f)

        trainset = data_cur_split["train"]
        testset = data_cur_split["test"]
        items_with_predictions = []

        for item in tqdm(testset, total=len(testset)):
            shots = get_shots(trainset, args.num_shots, args.multitask, args.language) if args.num_shots > 0 else {}
            prediction = None
            counter = 0
            while prediction is None and counter < args.generation_tries:
                if args.multitask:
                    prediction = generate_prediction_multitask(model, tokenizer, item.get("Article_title", None), item["Comment"], shots, device, args.use_logits, args.num_new_tokens, args.language)
                else:
                    prediction = generate_prediction(model, tokenizer, item.get("Article_title", None), item["Comment"], shots, device, args.use_logits, args.num_new_tokens, args.language)
                counter += 1
            if prediction is None:
                raise ValueError(f"Prediction is None after {args.generation_tries} tries.")
            item["prediction"] = prediction
            items_with_predictions.append(item)

        output_dir = args.path_predictions_dir
        os.makedirs(output_dir, exist_ok=True)
        path_fout = os.path.join(output_dir, f"{split}.json")
        print(f"Save predictions to {path_fout}")

        with open(path_fout, "w") as f:
            json.dump(items_with_predictions, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_splits_dir", type=str, required=True)
    parser.add_argument("--path_predictions_dir", type=str, required=True)
    parser.add_argument("--model_path_or_identifier", type=str, required=True)
    parser.add_argument("--num_new_tokens", type=int, default=1)
    parser.add_argument("--num_shots", type=int, default=0, choices=[0, 5])
    parser.add_argument("--multitask", action="store_true")
    parser.add_argument("--generation_tries", type=int, default=10, help="Number of tries to generate a valid prediction.")
    parser.add_argument("--use_logits", action="store_true", help="Use logits instead of generation to predict the label.")
    parser.add_argument("--language", choices=["en", "de"], help="Language of the prompt.")
    cmd_args = parser.parse_args()
    main(cmd_args)
