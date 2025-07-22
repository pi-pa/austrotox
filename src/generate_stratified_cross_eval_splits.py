import json
import os
import argparse
import math

import random
random.seed(42)


def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    return dataset


def split_dataset(dataset, num_splits):
    random.shuffle(dataset)
    split_size = math.ceil(len(dataset) / num_splits)
    splits = [dataset[i:i + split_size] for i in range(0, len(dataset), split_size)]
    return splits


def save_splits(splits, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, split in enumerate(splits):
        with open(os.path.join(output_dir, f'split_{i}.json'), 'w', encoding='utf-8') as file:
            json.dump(split, file, ensure_ascii=False, indent=4)


def create_data_versions(num_splits, path_splits_dir):
    data_versions = []
    for dev_split_idx in range(num_splits):
        train_splits = []
        dev_split = None
        test_split = None
        for i in range(num_splits):
            with open(os.path.join(path_splits_dir, f'split_{i}.json'), 'r', encoding='utf-8') as file:
                split_data = json.load(file)
            if i == dev_split_idx:
                dev_split = split_data
            elif i == (dev_split_idx + 1) % num_splits:
                test_split = split_data
            else:
                train_splits.extend(split_data)
        data_versions.append({'train': train_splits, 'dev': dev_split, 'test': test_split})
    return data_versions


def main():
    parser = argparse.ArgumentParser(description='Generate cross-validation splits for the dataset')
    parser.add_argument('--num_splits', type=int, help='Number of splits for cross-validation', default=10)
    parser.add_argument('--path_data', type=str, help='Path to the input dataset JSON file', default='data/all_comments.json')
    parser.add_argument('--path_splits_dir', type=str, help='Path to the directory to save splits', default='data/cross_eval_splits')
    parser.add_argument('--path_output_dir', type=str, help='Path to the directory to save data versions', default='data/train_dev_test')
    args = parser.parse_args()

    dataset = load_dataset(args.path_data)
    splits = split_dataset(dataset, args.num_splits)
    save_splits(splits, args.path_splits_dir)

    data_versions = create_data_versions(args.num_splits, args.path_splits_dir)
    os.makedirs(args.path_output_dir, exist_ok=True)
    for i, data_version in enumerate(data_versions):
        with open(os.path.join(args.path_output_dir, f'split_{i}.json'), 'w', encoding='utf-8') as file:
            json.dump(data_version, file, ensure_ascii=False, indent=4)
    print(f'Data versions generated and saved in {args.path_output_dir}')


if __name__ == '__main__':
    main()
