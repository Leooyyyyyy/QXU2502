import argparse
import csv
import json
import os
import re
from collections import Counter

try:
    import pandas as pd
except ImportError:
    pd = None


POSTURE_DIRS = ["downdog", "plank", "side_plank", "warrior_ii"]
POSTURE_NAMES = ["Down Dog", "Plank", "Side Plank", "Warrior II"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_DATASET_ROOT = "./dataset"
DEFAULT_SPLIT_ARTIFACT = "./artifacts/splits/refined_split_seed2024.json"


def debug_print(message):
    print("[debug] {0}".format(message))


def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def normalize_subtype_name(name):
    return re.sub(r"^\d+_", "", name.strip())


def build_dataset_entries(dataset_root):
    entries = []
    negative_subtypes_by_posture = dict((pose, set()) for pose in POSTURE_DIRS)

    for posture_idx, posture_dir in enumerate(POSTURE_DIRS):
        positive_path = os.path.join(dataset_root, posture_dir, "positive")
        if os.path.isdir(positive_path):
            for filename in sorted(os.listdir(positive_path)):
                if not is_image_file(filename):
                    continue
                entries.append({
                    "image_path": os.path.join(positive_path, filename),
                    "posture_idx": posture_idx,
                    "correctness": 1,
                    "negative_subtype": None,
                })

        negative_path = os.path.join(dataset_root, posture_dir, "negative")
        if not os.path.isdir(negative_path):
            continue

        has_subdirs = False
        for current_root, _, filenames in os.walk(negative_path):
            rel_root = os.path.relpath(current_root, negative_path)
            subtype = "unspecified_negative"
            if rel_root != ".":
                has_subdirs = True
                subtype = normalize_subtype_name(rel_root.split(os.sep)[0])

            for filename in sorted(filenames):
                if not is_image_file(filename):
                    continue
                negative_subtypes_by_posture[posture_dir].add(subtype)
                entries.append({
                    "image_path": os.path.join(current_root, filename),
                    "posture_idx": posture_idx,
                    "correctness": 0,
                    "negative_subtype": subtype,
                })

        if not has_subdirs:
            negative_subtypes_by_posture[posture_dir].add("unspecified_negative")

    original_entries = entries[:]
    flipped_entries = []
    for entry in original_entries:
        flipped_entry = dict(entry)
        flipped_entry["image_path"] = entry["image_path"] + " [flipped]"
        flipped_entries.append(flipped_entry)
    entries.extend(flipped_entries)

    metadata = {
        "posture_dirs": POSTURE_DIRS,
        "posture_names": POSTURE_NAMES,
        "negative_subtypes_by_posture": dict(
            (pose, sorted(subtypes) if subtypes else ["unspecified_negative"])
            for pose, subtypes in negative_subtypes_by_posture.items()
        ),
    }
    return entries, metadata


def label_to_class_name(entry, metadata):
    posture_name = metadata["posture_names"][entry["posture_idx"]]
    if entry["correctness"] == 1:
        return "{0} | Correct".format(posture_name)
    return "{0} | Incorrect | {1}".format(posture_name, entry["negative_subtype"])


def build_all_class_names(metadata):
    class_names = []
    for posture_name, posture_dir in zip(metadata["posture_names"], metadata["posture_dirs"]):
        class_names.append("{0} | Correct".format(posture_name))
        for subtype_name in metadata["negative_subtypes_by_posture"][posture_dir]:
            class_names.append("{0} | Incorrect | {1}".format(posture_name, subtype_name))
    return class_names


def load_split_artifact(split_artifact_path):
    with open(split_artifact_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_by_split(entries, indices, metadata):
    counter = Counter()
    for idx in indices:
        counter[label_to_class_name(entries[idx], metadata)] += 1
    return counter


def build_flags(total_count, train_count, val_count, test_count, split_metadata):
    flags = []

    if train_count == 0:
        flags.append("missing_train")
    if val_count == 0:
        flags.append("missing_val")
    if test_count == 0:
        flags.append("missing_test")
    if total_count < 30:
        flags.append("rare_total")

    if total_count >= 20:
        expected_ratios = {
            "train": split_metadata.get("train_ratio", 0.7),
            "val": split_metadata.get("val_ratio", 0.15),
            "test": split_metadata.get("test_ratio", 0.15),
        }
        observed_counts = {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        }
        for split_name in ("train", "val", "test"):
            expected_count = total_count * expected_ratios[split_name]
            observed_count = observed_counts[split_name]
            if abs(observed_count - expected_count) > max(2, total_count * 0.15):
                flags.append("{0}_imbalance".format(split_name))

    return ", ".join(flags)


def build_rows(entries, metadata, split_payload):
    all_class_names = build_all_class_names(metadata)
    total_counter = count_by_split(entries, range(len(entries)), metadata)
    train_counter = count_by_split(entries, split_payload["train_indices"], metadata)
    val_counter = count_by_split(entries, split_payload["val_indices"], metadata)
    test_counter = count_by_split(entries, split_payload["test_indices"], metadata)

    rows = []
    for class_name in all_class_names:
        total_count = total_counter.get(class_name, 0)
        train_count = train_counter.get(class_name, 0)
        val_count = val_counter.get(class_name, 0)
        test_count = test_counter.get(class_name, 0)
        rows.append({
            "class_name": class_name,
            "total_count": total_count,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "flags": build_flags(total_count, train_count, val_count, test_count, split_payload.get("metadata", {})),
        })
    return rows


def print_rows(rows):
    if pd is not None:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return

    headers = ["class_name", "total_count", "train_count", "val_count", "test_count", "flags"]
    widths = dict((header, len(header)) for header in headers)
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row[header])))

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))


def save_csv(rows, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "total_count", "train_count", "val_count", "test_count", "flags"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Report total/train/val/test counts for all classes from dataset folders and a saved split JSON.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Path to dataset root")
    parser.add_argument("--split-artifact", default=DEFAULT_SPLIT_ARTIFACT, help="Path to split artifact JSON")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path")
    args = parser.parse_args()

    print("script started")
    debug_print("dataset root path: {0}".format(args.dataset_root))
    debug_print("split artifact path: {0}".format(args.split_artifact))

    if not os.path.isdir(args.dataset_root):
        raise FileNotFoundError("Dataset root not found: {0}".format(args.dataset_root))
    if not os.path.exists(args.split_artifact):
        raise FileNotFoundError("Split artifact not found: {0}".format(args.split_artifact))

    entries, metadata = build_dataset_entries(args.dataset_root)
    split_payload = load_split_artifact(args.split_artifact)

    max_split_index = max(
        split_payload["train_indices"] + split_payload["val_indices"] + split_payload["test_indices"]
    )
    if max_split_index >= len(entries):
        raise ValueError(
            "Split indices reference {0} samples, but folder-derived dataset has only {1}. "
            "This usually means the split was created from a different dataset version.".format(
                max_split_index + 1,
                len(entries),
            )
        )

    rows = build_rows(entries, metadata, split_payload)

    debug_print("number of classes found: {0}".format(len(rows)))
    debug_print("number of samples in train split: {0}".format(len(split_payload["train_indices"])))
    debug_print("number of samples in val split: {0}".format(len(split_payload["val_indices"])))
    debug_print("number of samples in test split: {0}".format(len(split_payload["test_indices"])))
    print()
    print_rows(rows)

    missing_rows = [row for row in rows if "missing_" in row["flags"]]
    rare_rows = [row for row in rows if "rare_total" in row["flags"]]
    imbalance_rows = [row for row in rows if "imbalance" in row["flags"]]

    if missing_rows:
        print("\nClasses missing from at least one split:")
        print_rows(missing_rows)

    if rare_rows:
        print("\nRare classes:")
        print_rows(rare_rows)

    if imbalance_rows:
        print("\nClasses with suspicious split imbalance:")
        print_rows(imbalance_rows)

    if args.output_csv:
        save_csv(rows, args.output_csv)
        debug_print("saved csv: {0}".format(args.output_csv))


if __name__ == "__main__":
    main()
