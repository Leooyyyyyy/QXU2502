import json
import os
import pickle
from dataclasses import dataclass

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from stage3_utils import build_dataset_entries
from stage3_utils import normalize_keypoints

DATASET_BASE_PATH = "./dataset"
CACHE_PREFIX = "stage3_refined"
SPLIT_DIR = "./artifacts/splits"


@dataclass
class DatasetBundle:
    keypoints_np: np.ndarray
    labels: list[list[int]]
    image_paths: list[str]
    metadata: dict


class YogaPoseDataset(Dataset):
    def __init__(self, keypoints_np: np.ndarray, labels: list[list[int]]):
        self.keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.keypoints_tensor)

    def __getitem__(self, idx: int):
        return self.keypoints_tensor[idx], self.labels[idx]


def _cache_paths(cache_prefix: str) -> dict[str, str]:
    return {
        "keypoints": f"{cache_prefix}_blazepose_results.pkl",
        "labels": f"{cache_prefix}_label_list.pkl",
        "image_paths": f"{cache_prefix}_image_path_list.pkl",
        "metadata": f"{cache_prefix}_metadata.pkl",
    }


def load_or_build_refined_dataset(
    dataset_base_path: str = DATASET_BASE_PATH,
    cache_prefix: str = CACHE_PREFIX,
) -> DatasetBundle:
    cache_paths = _cache_paths(cache_prefix)

    if all(os.path.exists(path) for path in cache_paths.values()):
        with open(cache_paths["keypoints"], "rb") as f:
            keypoints_np = np.array(pickle.load(f))
        with open(cache_paths["labels"], "rb") as f:
            labels = pickle.load(f)
        with open(cache_paths["image_paths"], "rb") as f:
            image_paths = pickle.load(f)
        with open(cache_paths["metadata"], "rb") as f:
            metadata = pickle.load(f)
        print("Loaded cached refined dataset artifacts.")
        return DatasetBundle(keypoints_np=keypoints_np, labels=labels, image_paths=image_paths, metadata=metadata)

    entries, metadata = build_dataset_entries(dataset_base_path)

    image_list: list[np.ndarray] = []
    image_paths: list[str] = []
    labels: list[list[int]] = []

    for entry in entries:
        image = Image.open(entry["image_path"])
        image_np = np.array(image.convert("RGB"))
        image_list.append(image_np)
        image_paths.append(entry["image_path"])
        labels.append([entry["posture_idx"], entry["correctness"], entry["negative_subtype_idx"]])

    original_images = image_list[:]
    original_paths = image_paths[:]
    original_labels = labels[:]

    for image_np in original_images:
        image_list.append(np.fliplr(image_np))
    for image_path in original_paths:
        image_paths.append(image_path + " [flipped]")
    labels.extend(original_labels)

    blazepose_results: list[list[list[float]]] = []
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for image_np in image_list:
            result = pose.process(image_np)
            if not result.pose_landmarks:
                blazepose_results.append([])
                continue

            keypoints = []
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
            blazepose_results.append(keypoints)

    for idx in reversed(range(len(blazepose_results))):
        if blazepose_results[idx]:
            continue
        del blazepose_results[idx]
        del image_paths[idx]
        del labels[idx]

    keypoints_np = np.array(blazepose_results)
    for result_np in keypoints_np:
        normalize_keypoints(result_np)

    with open(cache_paths["keypoints"], "wb") as f:
        pickle.dump(keypoints_np, f)
    with open(cache_paths["labels"], "wb") as f:
        pickle.dump(labels, f)
    with open(cache_paths["image_paths"], "wb") as f:
        pickle.dump(image_paths, f)
    with open(cache_paths["metadata"], "wb") as f:
        pickle.dump(metadata, f)

    print("Built and cached refined dataset artifacts.")
    return DatasetBundle(keypoints_np=keypoints_np, labels=labels, image_paths=image_paths, metadata=metadata)


def split_artifact_path(split_name: str) -> str:
    os.makedirs(SPLIT_DIR, exist_ok=True)
    return os.path.join(SPLIT_DIR, f"{split_name}.json")


def save_split_indices(split_name: str, split_indices: dict[str, list[int]], split_metadata: dict) -> str:
    artifact_path = split_artifact_path(split_name)
    payload = {
        "split_name": split_name,
        "train_indices": split_indices["train"],
        "val_indices": split_indices["val"],
        "test_indices": split_indices["test"],
        "metadata": split_metadata,
    }
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return artifact_path


def load_split_indices(split_name: str) -> dict | None:
    artifact_path = split_artifact_path(split_name)
    if not os.path.exists(artifact_path):
        return None
    with open(artifact_path, "r", encoding="utf-8") as f:
        return json.load(f)
