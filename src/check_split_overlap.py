import json
import os
import argparse

def load_data_version(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data_version = json.load(file)
    return data_version

def stringify_dicts(dict_list):
    return [','.join(f"{k}:{v}" for k, v in d.items()) for d in dict_list]

def check_test_overlap(data_versions):
    test_splits_strings = [set(stringify_dicts(version['test'])) for version in data_versions]
    overlap = False
    for i, split1 in enumerate(test_splits_strings):
        for j, split2 in enumerate(test_splits_strings):
            if i != j and split1.intersection(split2):
                overlap = True
                print(f"Overlap found between test splits {i} and {j}")
    if not overlap:
        print("No overlap found between test splits.")


def main():
    parser = argparse.ArgumentParser(description='Check for overlap between test splits in different data versions')
    parser.add_argument('--num_versions', type=int, help='Number of data versions to check')
    parser.add_argument('--path_data_dir', type=str, help='Path to the directory containing data versions')
    args = parser.parse_args()

    data_versions = []
    for i in range(args.num_versions):
        data_version_path = os.path.join(args.path_data_dir, f'split_{i}.json')
        data_version = load_data_version(data_version_path)
        data_versions.append(data_version)

    check_test_overlap(data_versions)

if __name__ == '__main__':
    main()
