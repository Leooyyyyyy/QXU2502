import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from runners.sem2_train import YogaPoseDataset
from runners.sem2_train import count_correct_predictions
from runners.sem2_train import load_or_build_preprocessed_dataset
from runners.sem2_train import loss_func
from runners.sem2_train import outputs_to_pred_labels
from runners.sem2_train import split_dataset
from runners.sem2_train import label_to_confusion_name

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "three_stage_latest.pth"
OUTPUT_DIR = PROJECT_ROOT / "training_plots"


def save_current_figure(filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")


def plot_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 6), dpi=220)
    plt.plot(train_losses, linewidth=2.2, label="Training Loss")
    plt.plot(val_losses, linewidth=2.2, label="Validation Loss")
    plt.title("Training and Validation Loss", fontsize=18, pad=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=13)
    plt.tight_layout()
    save_current_figure("training_validation_loss.png")
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
    save_current_figure("training_validation_accuracy.png")
    plt.show()


def print_final_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")


def plot_confusion_matrix(model, metadata, test_loader, device, label_list):
    model.eval()
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

    for posture_idx, posture_name in enumerate(metadata["posture_names"]):
        posture_dir = metadata["posture_dirs"][posture_idx]
        posture_class_names = [f"{posture_name} | Correct"] + [
            f"{posture_name} | Incorrect | {subtype_name}"
            for subtype_name in metadata["negative_subtypes_by_posture"][posture_dir]
        ] + ["Other posture"]
        posture_class_to_idx = {name: idx for idx, name in enumerate(posture_class_names)}

        filtered_true_indexes = []
        filtered_pred_indexes = []

        for true_label, pred_label in zip(test_true_labels, test_pred_labels):
            true_posture_idx = true_label[0]
            pred_posture_idx = pred_label[0]
            if true_posture_idx != posture_idx:
                continue

            true_name = label_to_confusion_name(
                true_label,
                metadata["posture_names"],
                metadata["negative_subtypes_by_posture"],
                metadata["posture_dirs"],
            )
            filtered_true_indexes.append(posture_class_to_idx[true_name])

            if pred_posture_idx == posture_idx:
                pred_name = label_to_confusion_name(
                    pred_label,
                    metadata["posture_names"],
                    metadata["negative_subtypes_by_posture"],
                    metadata["posture_dirs"],
                )
                filtered_pred_indexes.append(posture_class_to_idx[pred_name])
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
        save_current_figure(f"{posture_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.show()


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = checkpoint["model"].to(device).eval()
    metadata = checkpoint["metadata"]

    print_final_metrics(
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        checkpoint["train_accuracies"],
        checkpoint["val_accuracies"],
    )

    plot_curves(
        checkpoint["train_losses"],
        checkpoint["val_losses"],
        checkpoint["train_accuracies"],
        checkpoint["val_accuracies"],
    )

    keypoints_np, label_list, _, _ = load_or_build_preprocessed_dataset()
    dataset = YogaPoseDataset(keypoints_np, label_list)
    _, _, test_dataset = split_dataset(dataset)
    test_loader = DataLoader(test_dataset, batch_size=32)

    plot_confusion_matrix(model, metadata, test_loader, device, label_list)


if __name__ == "__main__":
    main()
