import warnings
from time import time

import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from numpy import ndarray
from torch.nn import functional as F

from core.stage3_utils import normalize_keypoints
from core.stage3_utils import parse_true_label_from_path

warnings.filterwarnings("ignore")


class PostureCorrectionSystem:
    def __init__(self, checkpoint_path: str = "./checkpoints/three_stage_latest.pth"):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model = checkpoint["model"].to(self.device).eval()
        self.metadata = checkpoint["metadata"]
        self.posture_names = self.metadata["posture_names"]
        self.posture_dirs = self.metadata["posture_dirs"]
        self.negative_subtypes_by_posture = self.metadata["negative_subtypes_by_posture"]
        self.subtype_counts_by_posture = self.metadata["subtype_counts_by_posture"]

        self.blazepose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

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

    def second_stage(self, keypoints: ndarray) -> dict[str, np.ndarray]:
        inputs = torch.tensor([keypoints], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)

        posture_prob = F.softmax(outputs["posture_logits"][0], dim=0)
        correctness_prob = torch.sigmoid(outputs["correctness_logits"][0])

        negative_subtype_prob_by_posture = {}
        negative_subtype_logits = outputs["negative_subtype_logits"][0]
        for posture_idx, posture_dir in enumerate(self.posture_dirs):
            valid_count = self.subtype_counts_by_posture[posture_idx]
            valid_logits = negative_subtype_logits[posture_idx, :valid_count]
            negative_subtype_prob_by_posture[posture_dir] = F.softmax(valid_logits, dim=0).cpu().numpy()

        return {
            "posture_prob": posture_prob.cpu().numpy(),
            "correctness_prob": correctness_prob.cpu().numpy(),
            "negative_subtype_prob_by_posture": negative_subtype_prob_by_posture,
        }

    def process_image(self, image_np: ndarray, image_path: str = ""):
        keypoints = self.first_stage(image_np)

        if not keypoints:
            print("No BlazePose landmarks detected in the image.")
            return {"image_path": image_path, "status": "no_landmarks"}

        normalized_keypoints = normalize_keypoints(np.array(keypoints))
        stage_outputs = self.second_stage(normalized_keypoints)

        posture_prob = stage_outputs["posture_prob"]
        correctness_prob = stage_outputs["correctness_prob"]

        predicted_posture_idx = int(posture_prob.argmax())
        predicted_posture = self.posture_names[predicted_posture_idx]
        selected_correctness_prob = float(correctness_prob[predicted_posture_idx])
        predicted_feedback = "Correct" if selected_correctness_prob > 0.5 else "Incorrect"

        predicted_negative_subtype = None
        predicted_negative_subtype_prob = None
        negative_subtype_distribution = None

        if predicted_feedback == "Incorrect":
            posture_dir = self.posture_dirs[predicted_posture_idx]
            negative_subtype_distribution = stage_outputs["negative_subtype_prob_by_posture"][posture_dir]
            predicted_negative_subtype_idx = int(negative_subtype_distribution.argmax())
            predicted_negative_subtype = self.negative_subtypes_by_posture[posture_dir][predicted_negative_subtype_idx]
            predicted_negative_subtype_prob = float(negative_subtype_distribution[predicted_negative_subtype_idx])

        sorted_idx = np.argsort(posture_prob)[::-1]
        top1_idx = int(sorted_idx[0])
        top2_idx = int(sorted_idx[1])

        print(f"Posture classification probabilities: {posture_prob.tolist()}")
        print(f"Correctness probabilities: {correctness_prob.tolist()}")
        print(f"Predicted posture: {predicted_posture}")
        print(f"Predicted feedback: {predicted_feedback}")
        if predicted_negative_subtype is not None:
            print(f"Predicted negative subtype: {predicted_negative_subtype}")

        result = {
            "image_path": image_path,
            "status": "ok",
            "pred_posture_idx": predicted_posture_idx,
            "pred_posture": predicted_posture,
            "pred_feedback": predicted_feedback,
            "selected_correctness_prob": selected_correctness_prob,
            "pred_negative_subtype": predicted_negative_subtype,
            "pred_negative_subtype_prob": predicted_negative_subtype_prob,
            "top1_posture_prob": float(posture_prob[top1_idx]),
            "top2_posture": self.posture_names[top2_idx],
            "top2_posture_prob": float(posture_prob[top2_idx]),
            "posture_margin": float(posture_prob[top1_idx] - posture_prob[top2_idx]),
        }

        for posture_idx, posture_dir in enumerate(self.posture_dirs):
            posture_key = posture_dir.replace("side_plank", "side_plank").replace("warrior_ii", "warrior_ii")
            result[f"p_{posture_key}"] = float(posture_prob[posture_idx])
            result[f"c_{posture_key}"] = float(correctness_prob[posture_idx])

        if predicted_negative_subtype is not None:
            posture_dir = self.posture_dirs[predicted_posture_idx]
            for subtype_name, subtype_prob in zip(
                self.negative_subtypes_by_posture[posture_dir],
                negative_subtype_distribution,
            ):
                result[f"n_{posture_dir}_{subtype_name}"] = float(subtype_prob)

        return result


def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title(image_path)
    plt.show()


def main():
    image_paths = [
        "./dataset/downdog/positive/018.png",
        "./dataset/downdog/negative/0_alignment_issue/010.png",

        "./dataset/plank/positive/frame_71.png",
        "./dataset/plank/negative/0_alignment_issue/011.png",

        "./dataset/side_plank/positive/069.png",
        "./dataset/side_plank/negative/1_arm_issue/0100.png",

        "./dataset/warrior_ii/positive/0326.png",
        "./dataset/warrior_ii/negative/0_alignment_issue/frame_0073.png",
    ]

    image_list = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
        image_list.append(np.array(image_rgb))

    posture_system = PostureCorrectionSystem()
    rows = []

    start_time = time()
    for image_path, image_np in zip(image_paths, image_list):
        print(f"Processing image: '{image_path}'")

        true_posture, true_feedback, true_negative_subtype = parse_true_label_from_path(
            image_path,
            posture_names=posture_system.posture_names,
            posture_dirs=posture_system.posture_dirs,
        )
        result = posture_system.process_image(image_np, image_path=image_path)

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

        rows.append(result)
        print()

    end_time = time()
    print(f"Total time to process {len(image_paths)} images: {end_time - start_time:.2f} seconds.")

    df = pd.DataFrame(rows)
    df.to_csv("prediction_analysis.csv", index=False)
    print("Saved prediction analysis to prediction_analysis.csv")

    mis_df = df[df["is_fully_correct"] == 0].copy()
    mis_df.to_csv("misclassified_only.csv", index=False)
    print("Saved misclassified samples to misclassified_only.csv")

    del posture_system


if __name__ == "__main__":
    main()
