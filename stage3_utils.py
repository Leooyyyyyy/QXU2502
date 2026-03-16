import os
import re
from typing import Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler

POSTURE_DIRS = ["downdog", "plank", "side_plank", "warrior_ii"]
POSTURE_NAMES = ["Down Dog", "Plank", "Side Plank", "Warrior II"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
UNSPECIFIED_NEGATIVE_SUBTYPE = "unspecified_negative"
NO_NEGATIVE_SUBTYPE = "__none__"


def is_image_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMAGE_EXTENSIONS


def normalize_subtype_name(name: str) -> str:
    return re.sub(r"^\d+_", "", name.strip())


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    x, y, z, visibility = keypoints.T

    scaler = MinMaxScaler()
    x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()
    y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()

    z_norm = np.linalg.norm(z)
    if z_norm != 0:
        z /= z_norm

    return keypoints


def build_dataset_entries(dataset_base_path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    negative_subtypes_by_posture: dict[str, set[str]] = {pose: set() for pose in POSTURE_DIRS}

    for posture_idx, posture_dir in enumerate(POSTURE_DIRS):
        positive_path = os.path.join(dataset_base_path, posture_dir, "positive")
        if os.path.isdir(positive_path):
            for filename in sorted(os.listdir(positive_path)):
                if not is_image_file(filename):
                    continue

                entries.append({
                    "image_path": os.path.join(positive_path, filename),
                    "posture_idx": posture_idx,
                    "correctness": 1,
                    "negative_subtype": NO_NEGATIVE_SUBTYPE,
                })

        negative_path = os.path.join(dataset_base_path, posture_dir, "negative")
        if not os.path.isdir(negative_path):
            continue

        has_subdirs = False
        for current_root, _, filenames in os.walk(negative_path):
            rel_root = os.path.relpath(current_root, negative_path)
            subtype = UNSPECIFIED_NEGATIVE_SUBTYPE
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
            negative_subtypes_by_posture[posture_dir].add(UNSPECIFIED_NEGATIVE_SUBTYPE)

    ordered_negative_subtypes_by_posture: dict[str, list[str]] = {}
    negative_subtype_to_idx_by_posture: dict[str, dict[str, int]] = {}

    for posture_dir in POSTURE_DIRS:
        subtypes = sorted(negative_subtypes_by_posture[posture_dir])
        if not subtypes:
            subtypes = [UNSPECIFIED_NEGATIVE_SUBTYPE]

        ordered_negative_subtypes_by_posture[posture_dir] = subtypes
        negative_subtype_to_idx_by_posture[posture_dir] = {
            subtype: subtype_idx for subtype_idx, subtype in enumerate(subtypes)
        }

    for entry in entries:
        posture_dir = POSTURE_DIRS[entry["posture_idx"]]
        if entry["correctness"] == 1:
            entry["negative_subtype_idx"] = -1
        else:
            entry["negative_subtype_idx"] = negative_subtype_to_idx_by_posture[posture_dir][entry["negative_subtype"]]

    metadata = {
        "posture_dirs": POSTURE_DIRS,
        "posture_names": POSTURE_NAMES,
        "negative_subtypes_by_posture": ordered_negative_subtypes_by_posture,
        "max_negative_subtypes": max(len(subtypes) for subtypes in ordered_negative_subtypes_by_posture.values()),
        "num_postures": len(POSTURE_DIRS),
        "subtype_counts_by_posture": [
            len(ordered_negative_subtypes_by_posture[posture_dir]) for posture_dir in POSTURE_DIRS
        ],
    }

    return entries, metadata


def parse_true_label_from_path(
    image_path: str,
    posture_names: list[str] | None = None,
    posture_dirs: list[str] | None = None,
) -> tuple[str, str, str | None]:
    posture_names = posture_names or POSTURE_NAMES
    posture_dirs = posture_dirs or POSTURE_DIRS

    normalized_path = image_path.replace("\\", "/")
    path_lower = normalized_path.lower()

    true_posture = "Unknown"
    true_posture_dir = None
    for posture_dir, posture_name in zip(posture_dirs, posture_names):
        if f"/{posture_dir}/" in path_lower:
            true_posture = posture_name
            true_posture_dir = posture_dir
            break

    if "/positive/" in path_lower:
        return true_posture, "Correct", None

    negative_marker = "/negative/"
    if negative_marker not in path_lower:
        return true_posture, "Unknown", None

    suffix = normalized_path.split(negative_marker, maxsplit=1)[1]
    parts = [part for part in suffix.split("/") if part]
    if not parts:
        return true_posture, "Incorrect", UNSPECIFIED_NEGATIVE_SUBTYPE

    first_part = parts[0]
    if "." in first_part:
        return true_posture, "Incorrect", UNSPECIFIED_NEGATIVE_SUBTYPE

    return true_posture, "Incorrect", normalize_subtype_name(first_part)
