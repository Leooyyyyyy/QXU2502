import argparse
import json
import os
from collections import Counter

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from core import models
from core.experiment_configs import get_experiment_config
from core.experiment_eval import compute_loss
from core.experiment_eval import count_exact_matches
from core.experiment_eval import evaluate_model
from core.experiment_eval import save_metrics_artifacts
from core.experiment_eval import save_per_posture_confusion_matrices
from core.experiment_eval import save_training_curves
from core.experiment_splits import build_dataset_subsets
from core.experiment_splits import create_or_load_fixed_split
from core.yoga_data import YogaPoseDataset
from core.yoga_data import load_or_build_refined_dataset

ARTIFACTS_ROOT = "./artifacts/experiments"


def get_device():
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def build_weighted_sampler(train_dataset):
    train_labels = [tuple(label.tolist()) for _, label in train_dataset]
    label_counts = Counter(train_labels)
    weights = [label_counts[label_tuple] ** -1 for label_tuple in train_labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def _inverse_frequency_weights(counts: Counter, ordered_keys: list[int]) -> list[float]:
    total_count = sum(counts.values())
    num_classes = len(ordered_keys)
    weights = []
    for key in ordered_keys:
        count = counts[key]
        weights.append(total_count / (num_classes * count))
    return weights


def build_criterion_overrides(train_dataset, metadata, device):
    posture_counts = Counter()
    correctness_counts = Counter()
    negative_counts_by_posture = {idx: Counter() for idx in range(metadata["num_postures"])}

    for _, label in train_dataset:
        posture_idx, correctness, negative_subtype_idx = label.tolist()
        posture_counts[posture_idx] += 1
        correctness_counts[correctness] += 1
        if correctness == 0:
            negative_counts_by_posture[posture_idx][negative_subtype_idx] += 1

    posture_weight = torch.tensor(
        _inverse_frequency_weights(posture_counts, list(range(metadata["num_postures"]))),
        dtype=torch.float32,
        device=device,
    )

    positive_count = correctness_counts[1]
    negative_count = correctness_counts[0]
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1)],
        dtype=torch.float32,
        device=device,
    )

    negative_weight_rows = []
    for posture_idx, subtype_names in enumerate(metadata["negative_subtypes_by_posture"].values()):
        subtype_indices = list(range(len(subtype_names)))
        subtype_counter = negative_counts_by_posture[posture_idx]
        for subtype_idx in subtype_indices:
            if subtype_counter[subtype_idx] == 0:
                subtype_counter[subtype_idx] = 1
        negative_weight_rows.append(_inverse_frequency_weights(subtype_counter, subtype_indices))

    return {
        "posture_loss": torch.nn.CrossEntropyLoss(weight=posture_weight),
        "correctness_loss": torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "negative_subtype_loss_weights": negative_weight_rows,
    }


def build_loss_overrides(train_dataset, metadata, device, use_class_weighted_loss):
    if not use_class_weighted_loss:
        return None

    weighted_losses = build_criterion_overrides(train_dataset, metadata, device)
    negative_weight_rows = weighted_losses.pop("negative_subtype_loss_weights")
    full_weight_rows = []
    max_negative_subtypes = metadata["max_negative_subtypes"]
    for posture_idx in range(metadata["num_postures"]):
        valid_count = metadata["subtype_counts_by_posture"][posture_idx]
        full_weight = torch.ones(max_negative_subtypes, dtype=torch.float32, device=device)
        full_weight[:valid_count] = torch.tensor(negative_weight_rows[posture_idx], dtype=torch.float32, device=device)
        full_weight_rows.append(full_weight)

    weighted_losses["negative_subtype_weight_rows"] = full_weight_rows
    return weighted_losses


def train_one_epoch(model, data_loader, optimizer, device, metadata, criterion_overrides):
    model.train()
    total_loss = 0.0
    total_exact_matches = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = compute_loss(outputs, labels, metadata["subtype_counts_by_posture"], criterion_overrides)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_exact_matches += count_exact_matches(outputs, labels, metadata["subtype_counts_by_posture"])

    return total_loss / len(data_loader), total_exact_matches / len(data_loader.dataset)


def evaluate_epoch(model, data_loader, device, metadata, criterion_overrides):
    model.eval()
    total_loss = 0.0
    total_exact_matches = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = compute_loss(outputs, labels, metadata["subtype_counts_by_posture"], criterion_overrides)

            total_loss += loss.item()
            total_exact_matches += count_exact_matches(outputs, labels, metadata["subtype_counts_by_posture"])

    return total_loss / len(data_loader), total_exact_matches / len(data_loader.dataset)


def save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def run_experiment(experiment_name: str):
    config = get_experiment_config(experiment_name)
    if config.use_geometric_features:
        raise NotImplementedError("Geometric feature experiments are not implemented yet. Start with phase1.1_baseline_refined.")

    dataset_bundle = load_or_build_refined_dataset()
    dataset = YogaPoseDataset(dataset_bundle.keypoints_np, dataset_bundle.labels)
    split_bundle = create_or_load_fixed_split(
        dataset_bundle.labels,
        split_name=config.split_name,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    train_dataset, val_dataset, test_dataset = build_dataset_subsets(dataset, split_bundle)

    train_loader_kwargs = {"batch_size": config.batch_size}
    if config.use_weighted_sampler:
        train_loader_kwargs["sampler"] = build_weighted_sampler(train_dataset)
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    device = get_device()
    model = models.ThreeStageSharedMLP(
        num_postures=dataset_bundle.metadata["num_postures"],
        max_negative_subtypes=dataset_bundle.metadata["max_negative_subtypes"],
        dropout_p=config.dropout_p,
    ).to(device)

    criterion_overrides = build_loss_overrides(
        train_dataset,
        dataset_bundle.metadata,
        device,
        config.use_class_weighted_loss,
    )

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_plateau_factor,
        patience=config.lr_plateau_patience,
    )

    output_dir = os.path.join(ARTIFACTS_ROOT, config.name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint_path = os.path.join(output_dir, "best_checkpoint.pth")
    last_checkpoint_path = os.path.join(output_dir, "last_checkpoint.pth")
    epochs_without_improvement = 0
    num_epochs = 0

    while scheduler.get_last_lr()[0] > config.min_learning_rate:
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, device, dataset_bundle.metadata, criterion_overrides
        )
        val_loss, val_accuracy = evaluate_epoch(
            model, val_loader, device, dataset_bundle.metadata, criterion_overrides
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        num_epochs += 1
        print(
            f"Epoch {num_epochs} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        checkpoint_payload = {
            "experiment_name": config.name,
            "num_epochs": num_epochs,
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "metadata": dataset_bundle.metadata,
            "split_name": config.split_name,
            "config": config.to_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        }
        save_checkpoint(last_checkpoint_path, checkpoint_payload)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = num_epochs
            epochs_without_improvement = 0
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_val_loss"] = best_val_loss
            save_checkpoint(best_checkpoint_path, checkpoint_payload)
        else:
            epochs_without_improvement += 1

        scheduler.step(val_loss)

        if config.use_early_stopping and epochs_without_improvement >= config.early_stopping_patience:
            print(f"Early stopping triggered at epoch {num_epochs}.")
            break

    best_checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    best_model = best_checkpoint["model"].to(device).eval()

    train_eval = evaluate_model(
        best_model,
        DataLoader(train_dataset, batch_size=config.batch_size),
        device,
        dataset_bundle.metadata,
        [dataset_bundle.image_paths[idx] for idx in train_dataset.indices],
        criterion_overrides,
    )
    val_eval = evaluate_model(
        best_model,
        DataLoader(val_dataset, batch_size=config.batch_size),
        device,
        dataset_bundle.metadata,
        [dataset_bundle.image_paths[idx] for idx in val_dataset.indices],
        criterion_overrides,
    )
    test_eval = evaluate_model(
        best_model,
        test_loader,
        device,
        dataset_bundle.metadata,
        [dataset_bundle.image_paths[idx] for idx in test_dataset.indices],
        criterion_overrides,
    )

    metrics = {
        "experiment_key": config.name,
        "experiment_name": config.display_name,
        "final_epoch": num_epochs,
        "best_epoch": best_checkpoint["best_epoch"],
        "best_val_loss": best_checkpoint["best_val_loss"],
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "test": test_eval["metrics"],
    }

    save_training_curves(output_dir, train_losses, val_losses, train_accuracies, val_accuracies)
    save_per_posture_confusion_matrices(output_dir, test_eval["true_labels"], test_eval["pred_labels"], dataset_bundle.metadata)
    save_metrics_artifacts(output_dir, metrics, test_eval["per_class_rows"], test_eval["prediction_rows"])

    print(f"Saved experiment artifacts to {output_dir}")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run a configured yoga classification experiment.")
    parser.add_argument(
        "--experiment",
        default="phase1.1_baseline_refined",
        help="Experiment name from experiment_configs.py",
    )
    args = parser.parse_args()
    run_experiment(args.experiment)


if __name__ == "__main__":
    main()
