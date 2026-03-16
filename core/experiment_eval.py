import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def mask_invalid_negative_logits(logits, posture_indices, subtype_counts_by_posture):
    masked_logits = logits.clone()
    for row_idx, posture_idx in enumerate(posture_indices.tolist()):
        valid_count = subtype_counts_by_posture[posture_idx]
        if valid_count < masked_logits.size(1):
            masked_logits[row_idx, valid_count:] = -1e9
    return masked_logits


def compute_loss(outputs, label_batch, subtype_counts_by_posture, criterion_overrides=None):
    criterion_overrides = criterion_overrides or {}
    posture_class = label_batch[:, 0]
    correctness = label_batch[:, 1].float()
    negative_subtype = label_batch[:, 2]

    posture_loss_fn = criterion_overrides.get("posture_loss")
    if posture_loss_fn is None:
        posture_loss_fn = torch.nn.CrossEntropyLoss()
    correctness_loss_fn = criterion_overrides.get("correctness_loss")
    if correctness_loss_fn is None:
        correctness_loss_fn = torch.nn.BCEWithLogitsLoss()
    negative_loss_fn = criterion_overrides.get("negative_subtype_loss")
    negative_weight_rows = criterion_overrides.get("negative_subtype_weight_rows")
    if negative_loss_fn is None and negative_weight_rows is None:
        negative_loss_fn = torch.nn.CrossEntropyLoss()

    posture_loss = posture_loss_fn(outputs["posture_logits"], posture_class)

    selected_correctness_logits = outputs["correctness_logits"].gather(1, posture_class.unsqueeze(1))
    correctness_loss = correctness_loss_fn(selected_correctness_logits, correctness.unsqueeze(1))

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
        if negative_weight_rows is not None:
            row_weights = torch.stack([negative_weight_rows[int(posture_idx)] for posture_idx in negative_postures.tolist()])
            log_probs = torch.nn.functional.log_softmax(selected_negative_logits, dim=1)
            sample_weights = row_weights.gather(1, negative_targets.unsqueeze(1)).squeeze(1)
            nll = -log_probs.gather(1, negative_targets.unsqueeze(1)).squeeze(1)
            negative_loss = (sample_weights * nll).mean()
        else:
            negative_loss = negative_loss_fn(selected_negative_logits, negative_targets)

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


def count_exact_matches(outputs, labels, subtype_counts_by_posture):
    pred_labels = outputs_to_pred_labels(outputs, subtype_counts_by_posture)
    return torch.sum(torch.all(pred_labels == labels, dim=1)).item()


def label_to_leaf_name(label, metadata):
    posture_idx, correctness, negative_subtype_idx = label
    posture_name = metadata["posture_names"][posture_idx]
    if correctness == 1:
        return f"{posture_name} | Correct"
    posture_dir = metadata["posture_dirs"][posture_idx]
    subtype_name = metadata["negative_subtypes_by_posture"][posture_dir][negative_subtype_idx]
    return f"{posture_name} | Incorrect | {subtype_name}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_training_curves(output_dir, train_losses, val_losses, train_accuracies, val_accuracies):
    _ensure_dir(output_dir)

    plt.figure(figsize=(12, 6), dpi=220)
    plt.plot(train_losses, linewidth=2.2, label="Training Loss")
    plt.plot(val_losses, linewidth=2.2, label="Validation Loss")
    plt.title("Training and Validation Loss", fontsize=18, pad=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_validation_loss.png"))
    plt.close()

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
    plt.savefig(os.path.join(output_dir, "training_validation_accuracy.png"))
    plt.close()


def save_per_posture_confusion_matrices(output_dir, true_labels, pred_labels, metadata):
    _ensure_dir(output_dir)

    for posture_idx, posture_name in enumerate(metadata["posture_names"]):
        posture_dir = metadata["posture_dirs"][posture_idx]
        posture_class_names = [f"{posture_name} | Correct"] + [
            f"{posture_name} | Incorrect | {subtype_name}"
            for subtype_name in metadata["negative_subtypes_by_posture"][posture_dir]
        ] + ["Other posture"]
        posture_class_to_idx = {name: idx for idx, name in enumerate(posture_class_names)}

        filtered_true_indexes = []
        filtered_pred_indexes = []

        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label[0] != posture_idx:
                continue

            filtered_true_indexes.append(posture_class_to_idx[label_to_leaf_name(true_label, metadata)])
            if pred_label[0] == posture_idx:
                filtered_pred_indexes.append(posture_class_to_idx[label_to_leaf_name(pred_label, metadata)])
            else:
                filtered_pred_indexes.append(posture_class_to_idx["Other posture"])

        conf_matrix = confusion_matrix(
            filtered_true_indexes,
            filtered_pred_indexes,
            labels=list(range(len(posture_class_names))),
        )

        fig_size = max(9, min(16, len(posture_class_names) * 1.6))
        fig, ax = plt.subplots(dpi=220, figsize=(fig_size, fig_size))
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=posture_class_names)
        disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
        ax.set_title(f"{posture_name} Confusion Matrix", fontsize=18, pad=16)
        ax.set_xlabel("Predicted Label", fontsize=13, labelpad=12)
        ax.set_ylabel("True Label", fontsize=13, labelpad=12)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)
        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
            tick_label.set_rotation_mode("anchor")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{posture_dir}.png"))
        plt.close()


def save_metrics_artifacts(output_dir, metrics, per_class_rows, prediction_rows):
    _ensure_dir(output_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_name", "support", "precision", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    pd.DataFrame(prediction_rows).to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)


def evaluate_model(model, data_loader, device, metadata, image_paths, criterion_overrides=None):
    model.eval()
    total_loss = 0.0
    total_exact_matches = 0
    true_labels = []
    pred_labels = []
    prediction_rows = []
    running_index = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = compute_loss(outputs, labels, metadata["subtype_counts_by_posture"], criterion_overrides)

            total_loss += loss.item()
            total_exact_matches += count_exact_matches(outputs, labels, metadata["subtype_counts_by_posture"])

            batch_pred_labels = outputs_to_pred_labels(outputs, metadata["subtype_counts_by_posture"])
            batch_true_labels = labels.tolist()
            batch_pred_labels_list = batch_pred_labels.tolist()

            true_labels.extend(batch_true_labels)
            pred_labels.extend(batch_pred_labels_list)

            for idx_in_batch in range(batch_size):
                dataset_index = running_index + idx_in_batch
                prediction_rows.append({
                    "image_path": image_paths[dataset_index],
                    "true_label": label_to_leaf_name(batch_true_labels[idx_in_batch], metadata),
                    "pred_label": label_to_leaf_name(batch_pred_labels_list[idx_in_batch], metadata),
                    "true_posture_idx": batch_true_labels[idx_in_batch][0],
                    "pred_posture_idx": batch_pred_labels_list[idx_in_batch][0],
                    "true_correctness": batch_true_labels[idx_in_batch][1],
                    "pred_correctness": batch_pred_labels_list[idx_in_batch][1],
                    "true_negative_subtype_idx": batch_true_labels[idx_in_batch][2],
                    "pred_negative_subtype_idx": batch_pred_labels_list[idx_in_batch][2],
                })
            running_index += batch_size

    avg_loss = total_loss / len(data_loader)
    exact_match_accuracy = total_exact_matches / len(data_loader.dataset)

    class_names = sorted({label_to_leaf_name(label, metadata) for label in true_labels + pred_labels})
    true_names = [label_to_leaf_name(label, metadata) for label in true_labels]
    pred_names = [label_to_leaf_name(label, metadata) for label in pred_labels]

    precisions, recalls, f1_scores, supports = precision_recall_fscore_support(
        true_names,
        pred_names,
        labels=class_names,
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_names,
        pred_names,
        labels=class_names,
        average="macro",
        zero_division=0,
    )

    per_class_rows = []
    for class_name, support, precision, recall, f1 in zip(class_names, supports, precisions, recalls, f1_scores):
        per_class_rows.append({
            "class_name": class_name,
            "support": int(support),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    metrics = {
        "loss": float(avg_loss),
        "exact_match_accuracy": float(exact_match_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
    }

    return {
        "metrics": metrics,
        "per_class_rows": per_class_rows,
        "prediction_rows": prediction_rows,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
    }
