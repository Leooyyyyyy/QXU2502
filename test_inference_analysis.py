import warnings
warnings.filterwarnings("ignore")

from time import time
from typing import Any

import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F


class PostureCorrectionSystem:
    POSTURE_NAMES = ['Down Dog', 'Plank', 'Side Plank', 'Warrior II']
    FEEDBACKS = ['Incorrect', 'Correct']

    def __init__(self):
        checkpoint_path = "./checkpoints/27.pth"

        self.device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

        self.blazepose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        self.model = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False
        )['model'].eval()

    def __del__(self):
        if hasattr(self, "blazepose") and self.blazepose is not None:
            self.blazepose.close()

    def first_stage(self, image_np: ndarray) -> list[list[float]]:
        keypoints = []
        result = self.blazepose.process(image_np)

        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        return keypoints

    def second_stage(self, keypoints: ndarray) -> tuple[ndarray, ndarray]:
        inputs = torch.tensor([keypoints], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output: torch.Tensor = self.model(inputs)[0]

        posture_logits, correctness_logits = output.T
        posture_prob = F.softmax(posture_logits, dim=0)
        correctness_prob = torch.sigmoid(correctness_logits)

        return posture_prob.cpu().numpy(), correctness_prob.cpu().numpy()

    @staticmethod
    def normalize_keypoints(keypoints: ndarray) -> ndarray:
        x, y, z, visibility = keypoints.T

        scaler = MinMaxScaler()
        x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()
        y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        z_norm = np.linalg.norm(z)
        if z_norm != 0:
            z /= z_norm

        return keypoints

    def process_image(self, image_np: ndarray, image_path: str = "") -> dict[str, Any]:
        keypoints = self.first_stage(image_np)

        if not keypoints:
            return {
                "image_path": image_path,
                "status": "no_landmarks",
            }

        keypoints_np = np.array(keypoints)
        normalized_keypoints = self.normalize_keypoints(keypoints_np)

        posture_prob, correctness_prob = self.second_stage(normalized_keypoints)

        predicted_idx = int(posture_prob.argmax())
        predicted_posture = self.POSTURE_NAMES[predicted_idx]
        predicted_feedback = self.FEEDBACKS[round(float(correctness_prob[predicted_idx]))]

        sorted_idx = np.argsort(posture_prob)[::-1]
        top1_idx = int(sorted_idx[0])
        top2_idx = int(sorted_idx[1])

        top1_prob = float(posture_prob[top1_idx])
        top2_prob = float(posture_prob[top2_idx])
        posture_margin = top1_prob - top2_prob

        return {
            "image_path": image_path,
            "status": "ok",
            "pred_posture_idx": predicted_idx,
            "pred_posture": predicted_posture,
            "pred_feedback": predicted_feedback,
            "selected_correctness_prob": float(correctness_prob[predicted_idx]),
            "top1_posture_prob": top1_prob,
            "top2_posture": self.POSTURE_NAMES[top2_idx],
            "top2_posture_prob": top2_prob,
            "posture_margin": posture_margin,
            "p_down_dog": float(posture_prob[0]),
            "p_plank": float(posture_prob[1]),
            "p_side_plank": float(posture_prob[2]),
            "p_warrior_ii": float(posture_prob[3]),
            "c_down_dog": float(correctness_prob[0]),
            "c_plank": float(correctness_prob[1]),
            "c_side_plank": float(correctness_prob[2]),
            "c_warrior_ii": float(correctness_prob[3]),
        }


def load_image_with_flip(original_image_path: str, is_flipped: int) -> np.ndarray:
    image = Image.open(original_image_path)
    image_rgb = image.convert("RGB")
    image_np = np.array(image_rgb)

    if int(is_flipped) == 1:
        image_np = np.fliplr(image_np)

    return image_np


def main():
    # Read test paths csv
    df_paths = pd.read_csv("test_dataset_paths.csv")

    posture_system = PostureCorrectionSystem()
    rows = []

    start_time = time()

    for _, row in df_paths.iterrows():
        original_image_path = row["original_image_path"]
        is_flipped = int(row["is_flipped"])

        true_posture_idx = int(row["true_posture_idx"])
        true_posture = row["true_posture"]
        true_feedback_idx = int(row["true_feedback_idx"])
        true_feedback = row["true_feedback"]

        image_np = load_image_with_flip(original_image_path, is_flipped)

        result = posture_system.process_image(image_np, image_path=original_image_path)

        result["original_image_path"] = original_image_path
        result["is_flipped"] = is_flipped
        result["true_posture_idx"] = true_posture_idx
        result["true_posture"] = true_posture
        result["true_feedback_idx"] = true_feedback_idx
        result["true_feedback"] = true_feedback

        if result.get("status") == "ok":
            result["is_posture_correct"] = int(result["pred_posture"] == true_posture)
            result["is_feedback_correct"] = int(result["pred_feedback"] == true_feedback)
            result["is_fully_correct"] = int(
                (result["pred_posture"] == true_posture)
                and (result["pred_feedback"] == true_feedback)
            )
        else:
            result["is_posture_correct"] = 0
            result["is_feedback_correct"] = 0
            result["is_fully_correct"] = 0

        rows.append(result)

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