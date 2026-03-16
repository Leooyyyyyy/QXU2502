import argparse
import os
import csv
from typing import Dict, List

DATASET_ROOT = "./dataset"
OUTPUT_CSV = "refined_dataset_counts.csv"

POSES = ["downdog", "plank", "side_plank", "warrior_ii"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def count_images_in_folder(folder_path: str) -> int:
    total = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if is_image_file(f):
                total += 1
    return total


def safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        [name for name in os.listdir(path) if not name.startswith(".")]
    )


def analyze_pose(dataset_root: str, pose: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    pose_path = os.path.join(dataset_root, pose)
    positive_path = os.path.join(pose_path, "positive")
    negative_path = os.path.join(pose_path, "negative")

    positive_count = 0
    negative_total = 0
    negative_rows: List[Dict[str, object]] = []

    if os.path.isdir(positive_path):
        positive_count = count_images_in_folder(positive_path)

    if os.path.isdir(negative_path):
        subtypes = safe_listdir(negative_path)
        for subtype in subtypes:
            subtype_path = os.path.join(negative_path, subtype)
            if os.path.isdir(subtype_path):
                subtype_count = count_images_in_folder(subtype_path)
                negative_total += subtype_count
                negative_rows.append({
                    "Pose": pose,
                    "Split": "Negative (-ve)",
                    "Subtype": subtype,
                    "Number of images": subtype_count,
                    "Negative total": "",
                    "Pose total": "",
                    "Note": ""
                })

    pose_total = positive_count + negative_total

    rows.append({
        "Pose": pose,
        "Split": "Positive (+ve)",
        "Subtype": "positive",
        "Number of images": positive_count,
        "Negative total": "",
        "Pose total": "",
        "Note": ""
    })

    rows.extend(negative_rows)

    rows.append({
        "Pose": pose,
        "Split": "Summary",
        "Subtype": "NEGATIVE_TOTAL",
        "Number of images": negative_total,
        "Negative total": negative_total,
        "Pose total": "",
        "Note": "Negative total"
    })

    rows.append({
        "Pose": pose,
        "Split": "Summary",
        "Subtype": "POSE_TOTAL",
        "Number of images": pose_total,
        "Negative total": "",
        "Pose total": pose_total,
        "Note": "Pose total"
    })

    return rows


def analyze_dataset(dataset_root: str) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []

    for pose in POSES:
        pose_path = os.path.join(dataset_root, pose)
        if not os.path.isdir(pose_path):
            print("Warning: pose folder not found -> {0}".format(pose_path))
            continue

        pose_rows = analyze_pose(dataset_root, pose)
        all_rows.extend(pose_rows)

    return all_rows


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "Pose", "Split", "Subtype", "Number of images",
        "Negative total", "Pose total", "Note"
    ]

    widths = {h: len(h) for h in headers}

    for row in rows:
        for h in headers:
            value = str(row.get(h, ""))
            if len(value) > widths[h]:
                widths[h] = len(value)

    def format_row(row_dict: Dict[str, object]) -> str:
        return " | ".join(
            str(row_dict.get(h, "")).ljust(widths[h]) for h in headers
        )

    separator = "-+-".join("-" * widths[h] for h in headers)

    print(format_row({h: h for h in headers}))
    print(separator)
    previous_pose = None
    for row in rows:
        current_pose = row.get("Pose", "")
        if previous_pose is not None and current_pose != previous_pose:
            print()
        print(format_row(row))
        previous_pose = current_pose


def write_csv(rows: List[Dict[str, object]], output_csv: str) -> None:
    headers = [
        "Pose", "Split", "Subtype", "Number of images",
        "Negative total", "Pose total", "Note"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        previous_pose = None
        for row in rows:
            current_pose = row.get("Pose", "")
            if previous_pose is not None and current_pose != previous_pose:
                writer.writerow({h: "" for h in headers})
            writer.writerow(row)
            previous_pose = current_pose


def main() -> None:
    parser = argparse.ArgumentParser(description="Count dataset images by pose and subtype.")
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Path to dataset root. Defaults to ./dataset",
    )
    parser.add_argument(
        "--output-csv",
        default=OUTPUT_CSV,
        help="CSV output path. Defaults to refined_dataset_counts.csv",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_root):
        print("Dataset root not found: {0}".format(args.dataset_root))
        return

    rows = analyze_dataset(args.dataset_root)
    print_table(rows)
    write_csv(rows, args.output_csv)

    print("\nSaved CSV to: {0}".format(args.output_csv))


if __name__ == "__main__":
    main()
