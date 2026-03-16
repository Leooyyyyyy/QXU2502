from dataclasses import asdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    split_name: str
    seed: int = 2024
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.00001
    dropout_p: float = 0.1
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 30
    min_learning_rate: float = 1e-5
    use_weighted_sampler: bool = False
    use_class_weighted_loss: bool = False
    use_early_stopping: bool = False
    early_stopping_patience: int = 30
    use_geometric_features: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# The registry order below matches the intended experiment stage order for the project:
# 1. baseline_refined
# 2. baseline_refined_earlystop
# 3.1 baseline_refined_weighted_loss
# 3.2 baseline_refined_weighted_sampler
# 3.3 baseline_refined_weighted_both
# 4.1 baseline_refined_weighted_loss_earlystop
# 4.2 baseline_refined_weighted_sampler_earlystop
# 4.3 baseline_refined_weighted_both_earlystop
# 5. baseline_refined_weighted_both_earlystop_geom
#
# Notes:
# - Stage 4 entries are explicit so you can compare early stopping on each Stage 3 strategy.
# - Stage 5 is registered now for planning consistency, but geometric features are still
#   not implemented in the training pipeline.
EXPERIMENTS: dict[str, ExperimentConfig] = {
    # Stage 1: refined dataset baseline with the current stable training setup.
    "baseline_refined": ExperimentConfig(
        name="baseline_refined",
        split_name="refined_split_seed2024",
    ),
    # Stage 2: same as the refined baseline, but with early stopping enabled.
    "baseline_refined_earlystop": ExperimentConfig(
        name="baseline_refined_earlystop",
        split_name="refined_split_seed2024",
        use_early_stopping=True,
    ),
    # Stage 3.1: only class-weighted loss, without weighted sampling.
    "baseline_refined_weighted_loss": ExperimentConfig(
        name="baseline_refined_weighted_loss",
        split_name="refined_split_seed2024",
        use_class_weighted_loss=True,
    ),
    # Stage 3.2: only weighted sampling, without class-weighted loss.
    "baseline_refined_weighted_sampler": ExperimentConfig(
        name="baseline_refined_weighted_sampler",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
    ),
    # Stage 3.3: weighted sampling + class-weighted loss together.
    "baseline_refined_weighted_both": ExperimentConfig(
        name="baseline_refined_weighted_both",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
        use_class_weighted_loss=True,
    ),
    # Stage 4.1: weighted-loss-only + early stopping.
    "baseline_refined_weighted_loss_earlystop": ExperimentConfig(
        name="baseline_refined_weighted_loss_earlystop",
        split_name="refined_split_seed2024",
        use_class_weighted_loss=True,
        use_early_stopping=True,
    ),
    # Stage 4.2: weighted-sampler-only + early stopping.
    "baseline_refined_weighted_sampler_earlystop": ExperimentConfig(
        name="baseline_refined_weighted_sampler_earlystop",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
        use_early_stopping=True,
    ),
    # Stage 4.3: weighted sampler + class-weighted loss + early stopping.
    "baseline_refined_weighted_both_earlystop": ExperimentConfig(
        name="baseline_refined_weighted_both_earlystop",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
        use_class_weighted_loss=True,
        use_early_stopping=True,
    ),
    # Backward-compatible alias for the previous Stage 4 name.
    "baseline_refined_weighted_earlystop": ExperimentConfig(
        name="baseline_refined_weighted_earlystop",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
        use_class_weighted_loss=True,
        use_early_stopping=True,
    ),
    # Stage 5: weighted-both + early stopping + future geometric features.
    # Registered now, but geometric features are not implemented yet.
    "baseline_refined_weighted_both_earlystop_geom": ExperimentConfig(
        name="baseline_refined_weighted_both_earlystop_geom",
        split_name="refined_split_seed2024",
        use_weighted_sampler=True,
        use_class_weighted_loss=True,
        use_early_stopping=True,
        use_geometric_features=True,
    ),
}


def get_experiment_config(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        valid_names = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment '{name}'. Valid experiments: {valid_names}")
    return EXPERIMENTS[name]
