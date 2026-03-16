import os
import pickle
import warnings
from collections import Counter

import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

import models
from stage3_utils import build_dataset_entries
from stage3_utils import normalize_keypoints

warnings.filterwarnings("ignore")

DATASET_BASE_PATH = "./dataset"
CACHE_PREFIX = "stage3"
CHECKPOINT_PATH = "./checkpoints/three_stage_latest.pth"


def load_or_build_preprocessed_dataset():
    cache_paths = {
        "keypoints": f"{CACHE_PREFIX}_blazepose_results.pkl",
        "labels": f"{CACHE_PREFIX}_label_list.pkl",
        "image_paths": f"{CACHE_PREFIX}_image_path_list.pkl",
        "metadata": f"{CACHE_PREFIX}_metadata.pkl",
    }

    if all(os.path.exists(path) for path in cache_paths.values()):
        with open(cache_paths["keypoints"], "rb") as f:
            keypoints = pickle.load(f)
        with open(cache_paths["labels"], "rb") as f:
            labels = pickle.load(f)
        with open(cache_paths["image_paths"], "rb") as f:
            image_paths = pickle.load(f)
        with open(cache_paths["metadata"], "rb") as f:
            metadata = pickle.load(f)
        print("Loaded cached stage-3 preprocessing artifacts.")
        return np.array(keypoints), labels, image_paths, metadata

    entries, metadata = build_dataset_entries(DATASET_BASE_PATH)
    image_list = []
    image_path_list = []
    label_list = []

    for entry in entries:
        image = Image.open(entry["image_path"])
        image_rgb = image.convert("RGB")
        image_np = np.array(image_rgb)

        image_list.append(image_np)
        image_path_list.append(entry["image_path"])
        label_list.append([entry["posture_idx"], entry["correctness"], entry["negative_subtype_idx"]])

    original_images = image_list[:]
    original_paths = image_path_list[:]
    original_labels = label_list[:]

    for image_np in original_images:
        image_list.append(np.fliplr(image_np))
    for image_path in original_paths:
        image_path_list.append(image_path + " [flipped]")
    label_list.extend(original_labels)

    blazepose_results = []
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        for image_np in tqdm(image_list, desc="Extracting BlazePose keypoints"):
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
        del image_path_list[idx]
        del label_list[idx]

    blazepose_results_np = np.array(blazepose_results)
    for result_np in blazepose_results_np:
        normalize_keypoints(result_np)

    with open(cache_paths["keypoints"], "wb") as f:
        pickle.dump(blazepose_results_np, f)
    with open(cache_paths["labels"], "wb") as f:
        pickle.dump(label_list, f)
    with open(cache_paths["image_paths"], "wb") as f:
        pickle.dump(image_path_list, f)
    with open(cache_paths["metadata"], "wb") as f:
        pickle.dump(metadata, f)

    print("Built and cached stage-3 preprocessing artifacts.")
    return blazepose_results_np, label_list, image_path_list, metadata


class YogaPoseDataset(Dataset):
    def __init__(self, keypoints_np, labels_):
        self.keypoints_tensor = torch.tensor(keypoints_np, dtype=torch.float32)
        self.labels = torch.tensor(labels_, dtype=torch.long)

    def __len__(self):
        return len(self.keypoints_tensor)

    def __getitem__(self, idx):
        return self.keypoints_tensor[idx], self.labels[idx]


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, seed=2024):
    labels = [tuple(label.tolist()) for label in dataset.labels]
    indices = np.arange(len(dataset))

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=(1 - train_ratio),
        random_state=seed,
        stratify=labels,
    )

    temp_labels = [labels[idx] for idx in temp_indices]
    val_ratio_within_temp = val_ratio / (1 - train_ratio)

    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(1 - val_ratio_within_temp),
        random_state=seed,
        stratify=temp_labels,
    )

    return (
        Subset(dataset, train_indices.tolist()),
        Subset(dataset, val_indices.tolist()),
        Subset(dataset, test_indices.tolist()),
    )


def mask_invalid_negative_logits(logits, posture_indices, subtype_counts_by_posture):
    masked_logits = logits.clone()
    for row_idx, posture_idx in enumerate(posture_indices.tolist()):
        valid_count = subtype_counts_by_posture[posture_idx]
        if valid_count < masked_logits.size(1):
            masked_logits[row_idx, valid_count:] = -1e9
    return masked_logits


def loss_func(outputs, label_batch, subtype_counts_by_posture):
    posture_class = label_batch[:, 0]
    correctness = label_batch[:, 1].float()
    negative_subtype = label_batch[:, 2]

    posture_loss = nn.CrossEntropyLoss()(outputs["posture_logits"], posture_class)

    selected_correctness_logits = outputs["correctness_logits"].gather(1, posture_class.unsqueeze(1))
    correctness_loss = nn.BCEWithLogitsLoss()(selected_correctness_logits, correctness.unsqueeze(1))

    negative_mask = correctness == 0
    negative_loss = torch.tensor(0.0, device=posture_class.device)
    if torch.any(negative_mask):
        negative_postures = posture_class[negative_mask]
        negative_targets = negative_subtype[negative_mask]
        negative_logits = outputs["negative_subtype_logits"][negative_mask]
        selected_negative_logits = negative_logits[
            torch.arange(len(negative_postures), device=posture_class.device),
            negative_postures,
        ]
        selected_negative_logits = mask_invalid_negative_logits(
            selected_negative_logits, negative_postures, subtype_counts_by_posture
        )
        negative_loss = nn.CrossEntropyLoss()(selected_negative_logits, negative_targets)

    return posture_loss + correctness_loss + negative_loss


def outputs_to_pred_labels(outputs, subtype_counts_by_posture):
    posture_logits = outputs["posture_logits"]
    correctness_logits = outputs["correctness_logits"]
    negative_logits = outputs["negative_subtype_logits"]

    pred_posture = posture_logits.argmax(dim=1)
    selected_correctness_logits = correctness_logits.gather(1, pred_posture.unsqueeze(1)).squeeze(1)
    pred_correctness = (torch.sigmoid(selected_correctness_logits) > 0.5).long()

    pred_negative_subtype = torch.full_like(pred_posture, -1)
    negative_mask = pred_correctness == 0
    if torch.any(negative_mask):
        negative_postures = pred_posture[negative_mask]
        selected_negative_logits = negative_logits[negative_mask][
            torch.arange(torch.sum(negative_mask), device=pred_posture.device),
            negative_postures,
        ]
        selected_negative_logits = mask_invalid_negative_logits(
            selected_negative_logits, negative_postures, subtype_counts_by_posture
        )
        pred_negative_subtype[negative_mask] = selected_negative_logits.argmax(dim=1)

    return torch.stack([pred_posture, pred_correctness, pred_negative_subtype], dim=1)


def count_correct_predictions(outputs, labels, subtype_counts_by_posture):
    pred_labels = outputs_to_pred_labels(outputs, subtype_counts_by_posture)
    return torch.sum(torch.all(pred_labels == labels, dim=1)).item()


def label_to_confusion_name(label, posture_names, negative_subtypes_by_posture, posture_dirs):
    posture_idx, correctness, negative_subtype_idx = label
    posture_name = posture_names[posture_idx]
    if correctness == 1:
        return f"{posture_name} | Correct"

    posture_dir = posture_dirs[posture_idx]
    negative_name = negative_subtypes_by_posture[posture_dir][negative_subtype_idx]
    return f"{posture_name} | Incorrect | {negative_name}"


def build_weighted_sampler(train_dataset):
    train_dataset_labels = [tuple(label.tolist()) for _, label in train_dataset]
    label_counts = Counter(train_dataset_labels)
    weights = [label_counts[label_tuple] ** -1 for label_tuple in train_dataset_labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def main():
    keypoints_np, label_list, image_path_list, metadata = load_or_build_preprocessed_dataset()

    dataset = YogaPoseDataset(keypoints_np, label_list)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    test_rows = []
    for idx in test_dataset.indices:
        image_path = image_path_list[idx]
        label = label_list[idx]
        posture_idx, correctness, negative_subtype_idx = label
        posture_dir = metadata["posture_dirs"][posture_idx]
        negative_subtype = None
        if correctness == 0:
            negative_subtype = metadata["negative_subtypes_by_posture"][posture_dir][negative_subtype_idx]

        test_rows.append({
            "original_index": idx,
            "image_path": image_path,
            "original_image_path": image_path.replace(" [flipped]", ""),
            "is_flipped": int(image_path.endswith(" [flipped]")),
            "true_posture_idx": posture_idx,
            "true_posture": metadata["posture_names"][posture_idx],
            "true_feedback_idx": correctness,
            "true_feedback": "Correct" if correctness == 1 else "Incorrect",
            "true_negative_subtype_idx": negative_subtype_idx,
            "true_negative_subtype": negative_subtype,
        })

    pd.DataFrame(test_rows).to_csv("test_dataset_paths.csv", index=False)
    print("Saved test_dataset_paths.csv")

    train_sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = models.ThreeStageSharedMLP(
        num_postures=metadata["num_postures"],
        max_negative_subtypes=metadata["max_negative_subtypes"],
        dropout_p=0.1,
    ).to(device)
    print("No. of params (including bias):", sum(p.numel() for p in model.parameters()))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=30)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    num_epochs = 0

    def save_checkpoint():
        torch.save({
            "num_epochs": num_epochs,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "metadata": metadata,
        }, CHECKPOINT_PATH)
        print(f'Saved checkpoint to "{CHECKPOINT_PATH}"')

    while scheduler.get_last_lr()[0] > 1e-5:
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {num_epochs + 1} train"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels, metadata["subtype_counts_by_posture"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += count_correct_predictions(outputs, labels, metadata["subtype_counts_by_posture"])

        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_correct / len(train_loader.dataset)

        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_func(outputs, labels, metadata["subtype_counts_by_posture"])

                val_loss += loss.item()
                val_correct += count_correct_predictions(outputs, labels, metadata["subtype_counts_by_posture"])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_correct / len(val_loader.dataset)

        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        num_epochs += 1
        print(
            f"Epoch {num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}"
        )

        scheduler.step(avg_val_loss)
        print("Next Epoch Learning Rate:", scheduler.get_last_lr()[0])

        if num_epochs % 20 == 0:
            save_checkpoint()

        if scheduler.get_last_lr()[0] <= 1e-5:
            break

    save_checkpoint()

    plt.figure(figsize=(12, 6), dpi=220)
    plt.plot(train_losses, linewidth=2.2, label="Training Loss")
    plt.plot(val_losses, linewidth=2.2, label="Validation Loss")
    plt.title("Training and Validation Loss", fontsize=18, pad=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6), dpi=220)
    plt.plot(train_accuracies, linewidth=2.2, label="Training Accuracy")
    plt.plot(val_accuracies, linewidth=2.2, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy", fontsize=18, pad=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.show()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = checkpoint["model"].to(device).eval()
    metadata = checkpoint["metadata"]

    test_loss = 0.0
    test_correct = 0
    test_true_labels = []
    test_pred_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_func(outputs, labels, metadata["subtype_counts_by_posture"])

            test_loss += loss.item()
            test_correct += count_correct_predictions(outputs, labels, metadata["subtype_counts_by_posture"])

            test_true_labels.extend(labels.tolist())
            test_pred_labels.extend(outputs_to_pred_labels(outputs, metadata["subtype_counts_by_posture"]).tolist())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")

    class_names = sorted({
        label_to_confusion_name(label, metadata["posture_names"], metadata["negative_subtypes_by_posture"], metadata["posture_dirs"])
        for label in label_list
    })
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    true_indexes = [
        class_to_idx[
            label_to_confusion_name(label, metadata["posture_names"], metadata["negative_subtypes_by_posture"], metadata["posture_dirs"])
        ]
        for label in test_true_labels
    ]
    pred_indexes = [
        class_to_idx[
            label_to_confusion_name(label, metadata["posture_names"], metadata["negative_subtypes_by_posture"], metadata["posture_dirs"])
        ]
        for label in test_pred_labels
    ]

    conf_matrix = confusion_matrix(true_indexes, pred_indexes, labels=list(range(len(class_names))))

    fig_size = max(14, min(24, len(class_names) * 0.9))
    fig, ax = plt.subplots(dpi=220, figsize=(fig_size, fig_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=60, colorbar=False, values_format="d")
    ax.set_title("Three-Stage Confusion Matrix on the Test Set", fontsize=18, pad=16)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
    ax.set_ylabel("True Label", fontsize=13, labelpad=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
