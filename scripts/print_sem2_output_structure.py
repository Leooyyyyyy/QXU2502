#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


def describe_output(outputs: object) -> None:
    print(f"forward() output python type: {type(outputs)}")

    if isinstance(outputs, torch.Tensor):
        print(f"single tensor shape: {tuple(outputs.shape)}")
        return

    if isinstance(outputs, (list, tuple)):
        print(f"multi-tensor container length: {len(outputs)}")
        for i, item in enumerate(outputs):
            if isinstance(item, torch.Tensor):
                print(f"[{i}] shape={tuple(item.shape)} dtype={item.dtype}")
            else:
                print(f"[{i}] non-tensor type={type(item)}")
        return

    if isinstance(outputs, dict):
        print(f"dict keys: {list(outputs.keys())}")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
            else:
                print(f"{key}: non-tensor type={type(value)}")
        return

    print("Unhandled output type.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Semester 2 model forward output structure.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/three_stage_latest.pth",
        help="Path to Semester 2 checkpoint (.pth).",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Make sure pickle can resolve project-level modules like `models`.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = checkpoint["model"].to(device).eval()
    metadata = checkpoint.get("metadata", {})

    dummy_input = torch.randn(args.batch_size, 33, 4, device=device)

    print(f"checkpoint: {checkpoint_path}")
    print(f"model class: {model.__class__.__name__}")
    print(f"input shape: {tuple(dummy_input.shape)}")
    print(f"input dtype: {dummy_input.dtype}")

    with torch.no_grad():
        outputs = model(dummy_input)

    describe_output(outputs)

    if isinstance(outputs, dict):
        posture_logits = outputs.get("posture_logits")
        correctness_logits = outputs.get("correctness_logits")
        negative_subtype_logits = outputs.get("negative_subtype_logits")

        if isinstance(posture_logits, torch.Tensor):
            print("semantics: posture_logits[batch, posture_idx]")
        if isinstance(correctness_logits, torch.Tensor):
            print("semantics: correctness_logits[batch, posture_idx]")
        if isinstance(negative_subtype_logits, torch.Tensor):
            print("semantics: negative_subtype_logits[batch, posture_idx, subtype_slot_idx]")

    if metadata:
        print(f"metadata.num_postures: {metadata.get('num_postures')}")
        print(f"metadata.max_negative_subtypes: {metadata.get('max_negative_subtypes')}")
        print(f"metadata.subtype_counts_by_posture: {metadata.get('subtype_counts_by_posture')}")
        print(f"metadata.posture_dirs: {metadata.get('posture_dirs')}")


if __name__ == "__main__":
    main()
