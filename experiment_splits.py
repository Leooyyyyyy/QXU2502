import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from yoga_data import load_split_indices
from yoga_data import save_split_indices


def _labels_for_stratification(labels: list[list[int]]) -> list[str]:
    return [f"{label[0]}_{label[1]}_{label[2]}" for label in labels]


def create_or_load_fixed_split(labels: list[list[int]], split_name: str, train_ratio: float, val_ratio: float, seed: int):
    saved_split = load_split_indices(split_name)
    if saved_split is not None:
        return {
            "train": saved_split["train_indices"],
            "val": saved_split["val_indices"],
            "test": saved_split["test_indices"],
            "metadata": saved_split["metadata"],
        }

    label_groups = _labels_for_stratification(labels)
    indices = np.arange(len(labels))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=label_groups,
    )

    temp_labels = [label_groups[idx] for idx in temp_indices]
    val_ratio_within_temp = val_ratio / (1 - train_ratio)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_within_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    split_indices = {
        "train": train_indices.tolist(),
        "val": val_indices.tolist(),
        "test": test_indices.tolist(),
    }
    split_metadata = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1 - train_ratio - val_ratio,
        "stratify_label": "posture_correctness_negative_subtype",
    }
    save_split_indices(split_name, split_indices, split_metadata)

    return {
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
        "metadata": split_metadata,
    }


def build_dataset_subsets(dataset, split_indices):
    return (
        Subset(dataset, split_indices["train"]),
        Subset(dataset, split_indices["val"]),
        Subset(dataset, split_indices["test"]),
    )
