import json
import os
import argparse
from collections import Counter

def load_data_version(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data_version = json.load(file)
    return data_version

def compute_set_stats(data_versions):
    for i, data_version in enumerate(data_versions):
        train_size = len(data_version['train'])
        dev_size = len(data_version['dev'])
        test_size = len(data_version['test'])
        
        train_labels = [example['Label'] for example in data_version['train']]
        dev_labels = [example['Label'] for example in data_version['dev']]
        test_labels = [example['Label'] for example in data_version['test']]
        
        train_label_distribution = dict(Counter(train_labels))
        dev_label_distribution = dict(Counter(dev_labels))
        test_label_distribution = dict(Counter(test_labels))
        
        print(f"Data split {i}:")
        print(f"Train Set Size: {train_size}")
        print(f"Train Set Label Distribution: {train_label_distribution}")
        print(f"Dev Set Size: {dev_size}")
        print(f"Dev Set Label Distribution: {dev_label_distribution}")
        print(f"Test Set Size: {test_size}")
        print(f"Test Set Label Distribution: {test_label_distribution}")
        print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description='Compute sizes and label distributions of train/dev/test sets')
    parser.add_argument('--num_splits', type=int, help='Number of data splits to compute statistics for', default=10)
    parser.add_argument('--path_data_dir', type=str, help='Path to the directory containing data splits', default='data/train_dev_test')
    args = parser.parse_args()

    data_versions = []
    for i in range(args.num_splits):
        data_version_path = os.path.join(args.path_data_dir, f'split_{i}.json')
        data_version = load_data_version(data_version_path)
        data_versions.append(data_version)

    compute_set_stats(data_versions)

if __name__ == '__main__':
    main()
