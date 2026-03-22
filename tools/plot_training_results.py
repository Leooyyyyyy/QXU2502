import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "artifacts" / "experiments"

# Allow torch.load to resolve checkpoints saved with older top-level module paths.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_checkpoint(experiment_dir: Path):
    best_checkpoint_path = experiment_dir / "best_checkpoint.pth"
    last_checkpoint_path = experiment_dir / "last_checkpoint.pth"

    best_checkpoint = None
    last_checkpoint = None

    if best_checkpoint_path.exists():
        best_checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    if last_checkpoint_path.exists():
        last_checkpoint = torch.load(last_checkpoint_path, map_location="cpu", weights_only=False)

    if best_checkpoint is None and last_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found in {experiment_dir}")

    return {
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_path": best_checkpoint_path if best_checkpoint is not None else None,
        "last_checkpoint": last_checkpoint,
        "last_checkpoint_path": last_checkpoint_path if last_checkpoint is not None else None,
    }


def save_square_curve(
    values_train,
    values_val,
    title,
    ylabel,
    output_path: Path,
    show_plot: bool,
):
    plt.figure(figsize=(7.2, 7.2), dpi=220)
    plt.plot(values_train, linewidth=2.0, label=f"Training {ylabel}")
    plt.plot(values_val, linewidth=2.0, label=f"Validation {ylabel}")
    plt.title(title, fontsize=16, pad=12)
    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    if ylabel == "Accuracy":
        plt.ylim(0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def save_combined_curve_figure(
    train_losses,
    val_losses,
    train_accuracies,
    val_accuracies,
    output_path: Path,
    show_plot: bool,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), dpi=220)

    axes[0].plot(train_losses, linewidth=2.0, label="Training Loss")
    axes[0].plot(val_losses, linewidth=2.0, label="Validation Loss")
    axes[0].set_title("Training and Validation Loss", fontsize=16, pad=10)
    axes[0].set_xlabel("Epochs", fontsize=13)
    axes[0].set_ylabel("Loss", fontsize=13)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=12)

    axes[1].plot(train_accuracies, linewidth=2.0, label="Training Accuracy")
    axes[1].plot(val_accuracies, linewidth=2.0, label="Validation Accuracy")
    axes[1].set_title("Training and Validation Accuracy", fontsize=16, pad=10)
    axes[1].set_xlabel("Epochs", fontsize=13)
    axes[1].set_ylabel("Accuracy", fontsize=13)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close()


def regenerate_training_plots(experiment_dir: Path, show_plot: bool):
    checkpoint_bundle = load_checkpoint(experiment_dir)
    best_checkpoint = checkpoint_bundle["best_checkpoint"]
    last_checkpoint = checkpoint_bundle["last_checkpoint"] or best_checkpoint

    print(f"Loaded curve checkpoint: {checkpoint_bundle['last_checkpoint_path'] or checkpoint_bundle['best_checkpoint_path']}")
    if checkpoint_bundle["best_checkpoint_path"] is not None:
        print(f"Loaded best checkpoint: {checkpoint_bundle['best_checkpoint_path']}")

    train_losses = last_checkpoint["train_losses"]
    val_losses = last_checkpoint["val_losses"]
    train_accuracies = last_checkpoint["train_accuracies"]
    val_accuracies = last_checkpoint["val_accuracies"]

    print(f"Final stopping epoch: {last_checkpoint.get('num_epochs', len(train_losses))}")
    if best_checkpoint is not None:
        print(f"Best epoch: {best_checkpoint.get('best_epoch', 'N/A')}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.4f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.4f}")

    save_square_curve(
        train_losses,
        val_losses,
        "Training and Validation Loss",
        "Loss",
        experiment_dir / "training_validation_loss.png",
        show_plot,
    )
    save_square_curve(
        train_accuracies,
        val_accuracies,
        "Training and Validation Accuracy",
        "Accuracy",
        experiment_dir / "training_validation_accuracy.png",
        show_plot,
    )
    save_combined_curve_figure(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        experiment_dir / "training_validation_combined.png",
        show_plot,
    )

    print(f"Saved square training plots in: {experiment_dir}")


def experiment_dirs_from_args(args):
    experiments_dir = Path(args.experiments_dir).resolve() if args.experiments_dir else EXPERIMENTS_DIR
    if args.all_experiments:
        return sorted(path for path in experiments_dir.iterdir() if path.is_dir())

    experiment_dir = experiments_dir / args.experiment
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment folder not found: {experiment_dir}")
    return [experiment_dir]


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate square-format training loss/accuracy plots from saved experiment checkpoints."
    )
    parser.add_argument(
        "--experiment",
        default="phase2_baseline_refined_earlystop",
        help="Experiment folder name inside artifacts/experiments",
    )
    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help="Regenerate the training plots for all experiment folders",
    )
    parser.add_argument(
        "--experiments-dir",
        default="",
        help="Optional path to an experiments directory. Defaults to project artifacts/experiments",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plots interactively in addition to saving them",
    )
    args = parser.parse_args()

    for experiment_dir in experiment_dirs_from_args(args):
        print()
        print(f"Processing experiment: {experiment_dir.name}")
        regenerate_training_plots(experiment_dir, show_plot=args.show)


if __name__ == "__main__":
    main()
