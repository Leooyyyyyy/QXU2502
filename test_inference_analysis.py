import warnings
from time import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from sem2_main import PostureCorrectionSystem
from stage3_utils import parse_true_label_from_path

warnings.filterwarnings("ignore")


def load_image_with_flip(original_image_path: str, is_flipped: int) -> np.ndarray:
    image = Image.open(original_image_path)
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)
    if int(is_flipped) == 1:
        image_np = np.fliplr(image_np)
    return image_np


def evaluate_row(
    posture_system: PostureCorrectionSystem,
    original_image_path: str,
    is_flipped: int,
    true_posture: str,
    true_feedback: str,
    true_negative_subtype: str | None,
) -> dict[str, Any]:
    image_np = load_image_with_flip(original_image_path, is_flipped)
    result = posture_system.process_image(image_np, image_path=original_image_path)

    result["original_image_path"] = original_image_path
    result["is_flipped"] = is_flipped
    result["true_posture"] = true_posture
    result["true_feedback"] = true_feedback
    result["true_negative_subtype"] = true_negative_subtype

    if result.get("status") == "ok":
        result["is_posture_correct"] = int(result["pred_posture"] == true_posture)
        result["is_feedback_correct"] = int(result["pred_feedback"] == true_feedback)
        result["is_negative_subtype_correct"] = int(
            (true_feedback != "Incorrect") or (result["pred_negative_subtype"] == true_negative_subtype)
        )
        result["is_fully_correct"] = int(
            (result["pred_posture"] == true_posture)
            and (result["pred_feedback"] == true_feedback)
            and ((true_feedback != "Incorrect") or (result["pred_negative_subtype"] == true_negative_subtype))
        )
    else:
        result["is_posture_correct"] = 0
        result["is_feedback_correct"] = 0
        result["is_negative_subtype_correct"] = 0
        result["is_fully_correct"] = 0

    return result


def main():
    df_paths = pd.read_csv("test_dataset_paths.csv")

    posture_system = PostureCorrectionSystem()
    rows = []
    start_time = time()

    for _, row in df_paths.iterrows():
        original_image_path = row["original_image_path"]
        is_flipped = int(row["is_flipped"])

        true_posture = row.get("true_posture")
        true_feedback = row.get("true_feedback")
        true_negative_subtype = row.get("true_negative_subtype")

        if pd.isna(true_posture) or pd.isna(true_feedback):
            true_posture, true_feedback, true_negative_subtype = parse_true_label_from_path(
                original_image_path,
                posture_names=posture_system.posture_names,
                posture_dirs=posture_system.posture_dirs,
            )
        elif pd.isna(true_negative_subtype):
            true_negative_subtype = None

        rows.append(
            evaluate_row(
                posture_system=posture_system,
                original_image_path=original_image_path,
                is_flipped=is_flipped,
                true_posture=true_posture,
                true_feedback=true_feedback,
                true_negative_subtype=true_negative_subtype,
            )
        )

    end_time = time()
    print(f"Processed {len(rows)} test samples in {end_time - start_time:.2f} seconds.")

    df_results = pd.DataFrame(rows)
    df_results.to_csv("test_prediction_analysis.csv", index=False)
    print("Saved test_prediction_analysis.csv")

    df_mis = df_results[df_results["is_fully_correct"] == 0].copy()
    df_mis.to_csv("test_misclassified_only.csv", index=False)
    print("Saved test_misclassified_only.csv")

    df_results_original = df_results[df_results["is_flipped"] == 0].copy()
    df_results_original.to_csv("test_prediction_analysis_original_only.csv", index=False)

    df_mis_original = df_results_original[df_results_original["is_fully_correct"] == 0].copy()
    df_mis_original.to_csv("test_misclassified_original_only.csv", index=False)
    print("Saved original-only analysis csv files")

    del posture_system


if __name__ == "__main__":
    main()
