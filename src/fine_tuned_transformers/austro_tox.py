import json
from typing import List, Any, Tuple, Dict

import datasets
from datasets import DatasetInfo, Split, SplitGenerator, Features, Sequence, Value


def prepare_ner_data(data: List[Any]) -> List[Any]:
    # TODO: Werden ner tags auf article title berechnet?
    """Prepare data for NER training. There are two types of labels: vulgarity and target."""
    print("Preparing data")
    prepared_data = []

    for item in data:
        tokens = item["Comment"].split()
        vulgarity_labels = ["O"] * len(tokens)
        target_labels = ["O"] * len(tokens)

        for tag in item["Tags"]:
            if tag["Tag"] == "Vulgarity":
                start_idx = item["Comment"].find(tag["Token"])
                start_idx = len(item["Comment"][:start_idx].split())
                end_idx = start_idx + len(tag["Token"].split())
                #vulgarity_labels[start_idx:end_idx] = ["B-VULGARITY"] + [
                #    "I-VULGARITY"
                #] * (len(tag["Token"].split()) - 1)
                vulgarity_labels[start_idx:end_idx] = ["VULGARITY"] * (len(tag["Token"].split()))
            elif "Target" in tag["Tag"]:
                start_idx = item["Comment"].find(tag["Token"])
                start_idx = len(item["Comment"][:start_idx].split())
                end_idx = start_idx + len(tag["Token"].split())

                #label = f"B-{tag['Tag'].upper()}"
                #target_labels[start_idx:end_idx] = [label] + [
                #    f"I-{tag['Tag'].upper()}"
                #] * (len(tag["Token"].split()) - 1)
                label = tag['Tag'].upper()  # we don't need a beginning label
                target_labels[start_idx:end_idx] = [label] * (len(tag["Token"].split()))

        item["tokens"] = tokens
        item["vulgarity_labels"] = vulgarity_labels
        item["target_labels"] = target_labels
        prepared_data.append(item)

    return prepared_data


class AustroToxDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")  # 4.0.0

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=f"english_tox_{split}",  # f"austro_tox_{split}",
            version=datasets.Version("1.0.0"),  #4.0.0
            description="EnglishTox dataset",  # AustroTox dataset
        )
        for split in range(10)
    ]

    def __init__(self, data_file_path: str, **kwargs):
        super().__init__(**kwargs)
        self.data_file_path = data_file_path

    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            features=Features(
                {
                    "index": Value("string"),  # Value("int32"),
                    #"article_title": Value("string"),
                    "comment": Value("string"),
                    "combined_text": Value("string"),
                    "labels": Value("int32"),
                    "tags": Sequence(
                        {"Tag": Value("string"), "Token": Value("string")}
                    ),
                    "tokens": Sequence(Value("string")),
                    "vulgarity_labels": Sequence(Value("string")),
                    "target_labels": Sequence(Value("string")),
                }
            )
        )

    def _split_generators(self, dl_manager):
        with open(self.data_file_path, encoding="utf-8") as f:
            data = json.load(f)

        prepared_data = {
            "train": prepare_ner_data(data["train"]),
            "dev": prepare_ner_data(data["dev"]),
            "test": prepare_ner_data(data["test"]),
        }

        print("Train size:", len(data['train']))
        print("Dev size:", len(data['dev']))
        print("Test size:", len(data['test']))

        return [
            SplitGenerator(
                name=Split.TRAIN, gen_kwargs={"data": prepared_data["train"]}
            ),
            SplitGenerator(
                name=Split.VALIDATION, gen_kwargs={"data": prepared_data["dev"]}
            ),
            SplitGenerator(name=Split.TEST, gen_kwargs={"data": prepared_data["test"]}),
        ]

    def _generate_examples(self, data: List[Any]) -> Tuple[int, Dict]:
        #prepared_data = prepare_ner_data(data)
        for id_i, item in enumerate(data):
            yield id_i, {
                "index": item["Index"],
                #"article_title": item["Article_title"],
                "comment": item["Comment"],
                #"combined_text": f"article title: {item['Article_title']} \tcomment: {item['Comment']}",
                "combined_text": item["Comment"],
                "labels": item["Label"],
                "tags": item["Tags"],
                "tokens": item["tokens"],
                "vulgarity_labels": item["vulgarity_labels"],
                "target_labels": item["target_labels"],
            }
