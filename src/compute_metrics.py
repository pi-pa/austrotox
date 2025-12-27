import argparse
from collections import defaultdict
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, precision_score, recall_score
from seqeval.metrics import classification_report

from transformers import AutoTokenizer


def extract_true_and_pred_labels(data):
    true_labels = []
    pred_labels = []
    counter = 0
    for item in data:
        if "prediction_str" in item:
            key = "prediction_str"
        elif "prediction" in item:
            key = "prediction"
        if item[key] in [0, 1]:
            true_labels.append(int(item["Label"]))
            pred_labels.append(int(item[key]))
        else:
            counter += 1
    if counter > 0:
        print(f"{counter} out of {len(data)} predictions are not 0 (int) or 1 (int).")
    return true_labels, pred_labels


def load_true_labels(path_true_labels):
    true_labels = {}
    with open(path_true_labels, "r") as f:
        data = json.load(f)
        for item in data["test"]:
            true_labels[item["Index"]] = int(item["Label"])
    return [true_labels[i] for i in sorted(true_labels.keys())]


def load_predictions(path_predictions):
    predictions = {}
    with open(path_predictions, "r") as f:
        data = json.load(f)
        for item in data:
            if "prediction_str" in item:
                key = "prediction_str"
            elif "prediction" in item:
                key = "prediction"
            try:
                predictions[item["Index"]] = int(item[key]["Label"])
            except:
                try:
                    if item[key] in [0, 1]:
                        predictions[item["Index"]] = item[key]
                    else:
                        import pdb; pdb.set_trace()
                except:
                    import pdb; pdb.set_trace()
    return [predictions[i] for i in sorted(predictions.keys())]


def load_predictions_file(path_predictions):
    with open(path_predictions, "r") as f:
        return json.load(f)


def compute_metrics(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels)
    if len(set(true_labels)) == 2 and len(set(predicted_labels)) == 2:
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        pos_labels = [1, 'VULG']
        neg_labels = [0, 'O']
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label in pos_labels and predicted_label in pos_labels:
                true_positives += 1
            elif true_label in pos_labels and predicted_label in neg_labels:
                false_negatives += 1
            elif true_label in neg_labels and predicted_label in pos_labels:
                false_positives += 1
            elif true_label in neg_labels and predicted_label in neg_labels:
                true_negatives += 1
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "precision_binary": precision_score(true_labels, predicted_labels, average="binary"),
            "recall_binary": recall_score(true_labels, predicted_labels, average="binary"),
            "f1_binary": f1_score(true_labels, predicted_labels, average="binary"),
            
            "micro_precision": precision_score(true_labels, predicted_labels, average="micro"),
            "micro_recall": recall_score(true_labels, predicted_labels, average="micro"),
            "micro_f1": f1_score(true_labels, predicted_labels, average="micro"),
            
            "macro_precision": precision_score(true_labels, predicted_labels, average="macro"),
            "macro_recall": recall_score(true_labels, predicted_labels, average="macro"),
            "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "distinct_true_labels": list(set(true_labels)),
            "distinct_pred_labels": list(set(predicted_labels)),
        }
    else:
        metrics = {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "macro_precision": precision_score(true_labels, predicted_labels, average="macro"),
            "macro_recall": recall_score(true_labels, predicted_labels, average="macro"),
            "macro_f1": f1_score(true_labels, predicted_labels, average="macro"),
            "weighted_precision": precision_score(true_labels, predicted_labels, average="weighted"),
            "weighted_recall": recall_score(true_labels, predicted_labels, average="weighted"),
            "weighted_f1": f1_score(true_labels, predicted_labels, average="weighted"),
            "micro_precision": precision_score(true_labels, predicted_labels, average="micro"),
            "micro_recall": recall_score(true_labels, predicted_labels, average="micro"),
            "micro_f1": f1_score(true_labels, predicted_labels, average="micro"),
            "distinct_true_labels": list(set(true_labels)),
            "distinct_pred_labels": list(set(predicted_labels)),
        }
    return metrics


def compute_toxicity_metrics(dataset):
    true_labels = []
    pred_labels = []
    counter = 0
    for entry in dataset:
        if "prediction" in entry:
            try:
                pred_labels.append(entry["prediction"]["Label"])
                true_labels.append(entry["Label"])
            except:
                counter += 1
        elif "prediction_str" in entry:
            pred_labels.append(entry["prediction_str"])
            true_labels.append(entry["Label"])
        else:
            try:
                import re
                match = re.search(r'"Label": (\d)', entry["prediction"])
                pred_labels.append(int(match.group(1)))
                true_labels.append(entry["Label"])
            except:
                import pdb; pdb.set_trace()
    print(f"{counter} out of {len(dataset)} could not be parsed.")
    return compute_metrics(true_labels, pred_labels)


def tags_to_bio_vulg(tags, tokens, tokenizer):
    bio_tags = ["O"] * len(tokens)  # Initialize with "Outside" tags
    for tag_info in tags:
        if tag_info["Tag"] == "Vulgarity":
            subtokens = tokenizer.tokenize(tag_info["Token"])
            for st in subtokens:
                try:
                    subtoken_index = tokens.index(st)
                    bio_tags[subtoken_index] = "VULG"
                except:
                    pass
    return bio_tags


def tags_to_bio_target(tags, tokens, tokenizer):
    bio_tags = ["O"] * len(tokens)  # Initialize with "Outside" tags
    for tag_info in tags:
        if tag_info["Tag"] == "Target_Group":
            subtokens = tokenizer.tokenize(tag_info["Token"])
            for st in subtokens:
                try:
                    subtoken_index = tokens.index(st)
                    bio_tags[subtoken_index] = "GROUP"
                    break
                except:
                    pass
        elif tag_info["Tag"] == "Target_Individual":
            subtokens = tokenizer.tokenize(tag_info["Token"])
            for st in subtokens:
                try:
                    subtoken_index = tokens.index(st)
                    bio_tags[subtoken_index] = "INDIV"
                    break
                except:
                    pass
        elif tag_info["Tag"] == "Target_OTHER":
            subtokens = tokenizer.tokenize(tag_info["Token"])
            for st in subtokens:
                try:
                    subtoken_index = tokens.index(st)
                    bio_tags[subtoken_index] = "OTHER"
                    break
                except:
                    pass
    return bio_tags


def remove_non_span_tags(true_bio_tags, pred_bio_tags, requirement):
    """Remove all tags that are not span tags.

    Args:
        true_bio_tags (list of str): ground truth
        pred_bio_tags (list of str): predictions
        requirement (str): either "both", "true", or "pred"

    Returns:
        _type_: _description_
    """
    assert requirement in ["both", "true", "pred"]
    assert len(true_bio_tags) == len(pred_bio_tags)
    true_bio_span_tags = []
    pred_bio_span_tags = []
    for i in range(len(true_bio_tags)):
        if requirement == "both":
            if true_bio_tags[i] == "O" and pred_bio_tags[i] == "O":
                continue
            else:
                true_bio_span_tags.append(true_bio_tags[i])
                pred_bio_span_tags.append(pred_bio_tags[i])
        elif requirement == "true":
            if true_bio_tags[i] == "O":
                continue
            else:
                true_bio_span_tags.append(true_bio_tags[i])
                pred_bio_span_tags.append(pred_bio_tags[i])
        elif requirement == "pred":
            if pred_bio_tags[i] == "O":
                continue
            else:
                true_bio_span_tags.append(true_bio_tags[i])
                pred_bio_span_tags.append(pred_bio_tags[i])
    return true_bio_span_tags, pred_bio_span_tags


def compute_vulgarity_metrics(dataset, tokenizer, consider_only_spans, span_requirement):
    true_labels = []
    pred_labels = []

    # Iterate over all entries in the dataset
    for entry in dataset:

        # Tokenize the comment text
        tokens = tokenizer.tokenize(entry["Comment"])

        # Convert the ground truth and predicted tags to BIO format
        true_bio_tags = tags_to_bio_vulg(entry["Tags"], tokens, tokenizer)
        try:
            pred_bio_tags = tags_to_bio_vulg(entry["prediction"]["Tags"], tokens, tokenizer)
        except:
            pred_bio_tags = None
            print(f'Warning: no vulgarity prediction for entry {entry["Index"]}')
            continue
        
        if consider_only_spans: # remove all tags that are not vulgarity
            true_bio_tags, pred_bio_tags = remove_non_span_tags(true_bio_tags, pred_bio_tags, requirement=span_requirement)

        # Append to the overall list
        if pred_bio_tags:
            true_labels.append(true_bio_tags)
            pred_labels.append(pred_bio_tags)

    # Flatten the lists
    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_pred_labels = [label for sublist in pred_labels for label in sublist]
    # convert to 1/0 labels
    flat_true_labels_int = [1 if label == "VULG" else 0 for label in flat_true_labels]
    flat_pred_labels_int = [1 if label == "VULG" else 0 for label in flat_pred_labels]
    return compute_metrics(flat_true_labels_int, flat_pred_labels_int)


def compute_target_metrics(dataset, tokenizer, consider_only_spans, span_requirement):
    true_labels = []
    pred_labels = []
    
    for entry in dataset:
        tokens = tokenizer.tokenize(entry["Comment"])
        true_bio_tags = tags_to_bio_target(entry["Tags"], tokens, tokenizer)
        try:
            pred_bio_tags = tags_to_bio_target(entry["prediction"]["Tags"], tokens, tokenizer)
        except:
            pred_bio_tags = None
            print(f'Warning: no target prediction for entry {entry["Index"]}')
            continue
        if consider_only_spans:
            true_bio_tags, pred_bio_tags = remove_non_span_tags(true_bio_tags, pred_bio_tags, requirement=span_requirement)
        if pred_bio_tags:
            true_labels.append(true_bio_tags)
            pred_labels.append(pred_bio_tags)
    
    flat_true_labels = [label for sublist in true_labels for label in sublist]
    flat_pred_labels = [label for sublist in pred_labels for label in sublist]
    return compute_metrics(flat_true_labels, flat_pred_labels)


def main(args):
    for split in os.listdir(args.path_true_labels):
        split = split.split(".")[0]
        print(20*"=" + f" Computeing metrics for {split} " + 20*"=")
        path_true_labels = os.path.join(args.path_true_labels, f"{split}.json")
        path_predictions = os.path.join(args.path_predictions, f"{split}.json") # testset_predictions_
        if args.consider_only_spans:
            path_metrics = os.path.join(args.path_metrics_dir, f"{split}_only_spans_{args.span_requirement}.json")
        else:
            path_metrics = os.path.join(args.path_metrics_dir, f"{split}.json")
        os.makedirs(os.path.dirname(path_metrics), exist_ok=True)
        if args.multitask:
            dataset = load_predictions_file(path_predictions)
            tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
            toxicity_metrics = compute_toxicity_metrics(dataset)
            vulgarity_metrics = compute_vulgarity_metrics(dataset, tokenizer, consider_only_spans=args.consider_only_spans, span_requirement=args.span_requirement)
            target_metrics = compute_target_metrics(dataset, tokenizer, consider_only_spans=args.consider_only_spans, span_requirement=args.span_requirement)
            metrics = {"toxicity": toxicity_metrics, "vulgarity": vulgarity_metrics, "target": target_metrics}
        else:
            # if only binary toxicity is predicted
            # true_labels = load_true_labels(path_true_labels)
            # predictions = load_predictions(path_predictions)
            data = load_predictions_file(path_predictions)
            true_labels, predictions = extract_true_and_pred_labels(data)
            metrics = compute_metrics(true_labels, predictions)
        with open(path_metrics, "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        print(f"Metrics saved at {path_metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_predictions", type=str, required=True, help="Path to predictions directory containing split-directories.")
    parser.add_argument("--path_true_labels", type=str, required=True, help="Path to train-dev-test-file.")
    parser.add_argument("--path_metrics_dir", type=str, required=True, help="Path to metrics directory, continaing split directories.")
    parser.add_argument("--multitask", action="store_true", help="Whether to compute metrics for multitask model.")
    parser.add_argument("--consider_only_spans", action="store_true", help="Whether to consider only span tags for computing metrics.")
    parser.add_argument("--span_requirement", type=str, choices=["both", "true", "pred"], 
                        help="What counts as a span: ground truth spans, predicted spans, or spans that are both ground truth and predicted.")
    cmd_args = parser.parse_args()
    main(cmd_args)
