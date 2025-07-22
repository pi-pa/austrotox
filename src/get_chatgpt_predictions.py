import argparse
import json
import os
import random
import time
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import openai
from prompts import ChatGPTPrompts


with open('../openai_api_key_cl_uzh.txt') as fin:
    openai.api_key = fin.read().strip()


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def get_shots(data, n, multitask, language):
    """Sample n random instances from the given data."""
    samples = random.sample(data, n)
    sample_dict = {}
    for i in range(n):
        if language == 'de':
            sample_dict[f"title_{i+1}"] = samples[i]['Article_title']
        sample_dict[f"comment_{i+1}"] = samples[i]['Comment']
        if multitask:
            sample_dict[f"response_{i+1}"] = json.dumps({"Label": samples[i]["Label"], "Tags": samples[i]['Tags']}, ensure_ascii=False)
        else:
            sample_dict[f"response_{i+1}"] = json.dumps({"Label": samples[i]["Label"]}, ensure_ascii=False)
    return sample_dict

def predict_toxicity(sysem_msg: str, user_msg: str, model_name: str) -> str:
    messages=[{"role": "system", "content": sysem_msg}, {"role": "user", "content": user_msg}]
    try:
        response = completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=20,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        print(f'Exception {e}')
        import pdb; pdb.set_trace()
    try:
        prediction = json.loads(response.choices[0]['message']['content'])['Label']
        return prediction
    except:
        import pdb; pdb.set_trace()


def predict_multitask(sysem_msg: str, user_msg: str, model_name: str) -> str:
    messages=[{"role": "system", "content": sysem_msg}, {"role": "user", "content": user_msg}]
    try:
        response = completion_with_backoff(
            model=model_name,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"}
        )
    except Exception as e:
        print(f'Exception {e}')
        try:
            response = completion_with_backoff(
                model=model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            print(f'Exception {e}')
            return None
    try:
        json_prediction = json.loads(response.choices[0]['message']['content'])
        return json_prediction
    except:
        try:
            prediction = response.choices[0]['message']['content']
            return prediction
        except:
            print(f'Unexpected response: {response.choices[0]}')
            return None



def predict(item, train_data, multitask, num_shots, model_name, language):
    if num_shots == 0:
        if multitask:
            prediction = predict_multitask(
                ChatGPTPrompts.get_multitask_system_msg(num_shots, {}, language),
                ChatGPTPrompts.get_user_msg(title=item.get('Article_title', None), comment=item['Comment'], num_shots=num_shots, language=language),
                model_name=model_name
            )
        else:
            prediction = predict_toxicity(
                ChatGPTPrompts.get_toxicity_system_msg(num_shots, {}, language), 
                ChatGPTPrompts.get_user_msg(title=item['Article_title'], comment=item['Comment'], num_shots=num_shots, language=language),
                model_name=model_name
            )
    elif num_shots == 5:
        shots = get_shots(train_data, num_shots, multitask, language)
        if multitask:
            prediction = predict_multitask(
                ChatGPTPrompts.get_multitask_system_msg(num_shots, shots, language),
                ChatGPTPrompts.get_user_msg(title=item.get('Article_title', None), comment=item['Comment'], num_shots=num_shots, language=language),
                model_name=model_name
            )
        else:
            prediction = predict_toxicity(
                ChatGPTPrompts.get_toxicity_system_msg(num_shots, shots, language), 
                ChatGPTPrompts.get_user_msg(title=item.get('Article_title', None), comment=item['Comment'], num_shots=num_shots, language=language),
                model_name=model_name
            )
    else:
        raise ValueError(f'Unexpected number of shots: {num_shots}')
    return prediction


def main(args):
    # make sure that output directory exists
    out_dir = os.path.dirname(args.output_path)
    os.makedirs(out_dir, exist_ok=True)
    
    # one input and one output file
    if os.path.isfile(args.input_path):
        assert args.output_path.endswith('.json')
        predictions = []
        with open(args.input_path) as fin:
            print(f"Processing {args.input_path}")
            data = json.load(fin)
            train_data = data['train']
            test_data = data['test']
            for i, item in enumerate(tqdm(test_data)):
                prediction = predict(
                    item=item, 
                    train_data=train_data, 
                    multitask=args.multitask, 
                    num_shots=args.num_shots, 
                    model_name=args.model_name
                )
                item['prediction'] = prediction
                predictions.append(item)
                # time.sleep(1)
        
        print(f"Write predictions to {args.output_path}")
        with open(args.output_path, 'w') as fout:
            json.dump(predictions, fout, indent=4, ensure_ascii=False)
    
    # multiple input and output files
    elif os.path.isdir(args.input_path):
        if os.path.exists(args.output_path):
            # if it is not empty, raise an error
            if os.listdir(args.output_path):
                raise ValueError(f'Output directory {args.output_path} is not empty.')
        else:
            os.makedirs(args.output_path, exist_ok=True)
        for fname in os.listdir(args.input_path):
            with open(os.path.join(args.input_path, fname)) as fin:
                print(f"Processing {os.path.join(args.input_path, fname)}")
                data = json.load(fin)
                train_data = data['train']
                test_data = data['test']
                predictions = []
                for i, item in enumerate(tqdm(test_data)):
                    prediction = predict(
                        item=item, 
                        train_data=train_data, 
                        multitask=args.multitask, 
                        num_shots=args.num_shots, 
                        model_name=args.model_name,
                        language=args.language
                    )
                    item['prediction'] = prediction
                    predictions.append(item)
                    # time.sleep(1)

            print(f"Write predictions to {os.path.join(args.output_path, fname)}")
            with open(os.path.join(args.output_path, fname), 'w') as fout:
                json.dump(predictions, fout, indent=4, ensure_ascii=False)

    else:
        raise ValueError(f'Invalid input path: {args.input_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='Path to input file or directory.')
    parser.add_argument('--output_path', help='Path to output file or directory.')
    parser.add_argument('--multitask', action='store_true', help='Use multitask model.')
    parser.add_argument('--num_shots', type=int, choices=[0, 5], help='Number of examples provided in prompt.')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--model_name', type=str, choices=['gpt-3.5-turbo-1106', 'gpt-4-1106-preview'])
    parser.add_argument('--language', choices=['de', 'en'], help='Language of the prompt.')
    cmd_args = parser.parse_args()
    random.seed(cmd_args.random_seed)
    main(cmd_args)
