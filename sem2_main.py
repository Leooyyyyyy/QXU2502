# %% Import necessary modules
import warnings

# Suppress all Python warnings (global)
warnings.filterwarnings("ignore")

from time import time

import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from PIL import Image
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


# %% Posture Correction System Class
class PostureCorrectionSystem:
    POSTURE_NAMES = ['Down Dog', 'Plank', 'Side Plank', 'Warrior II']
    FEEDBACKS = ['Incorrect', 'Correct']

    def __init__(self):
        """
        Initializes the posture correction system by loading the necessary models and setting the device.
        """
        checkpoint_path = "./checkpoints/27.pth"

        # Set the device based on availability
        self.device = torch.device(
            'mps' if torch.backends.mps.is_available()
            else 'cuda' if torch.cuda.is_available()
            else 'cpu'
        )

        # Load the BlazePose model for keypoint detection
        self.blazepose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

        # Load the pre-trained neural network model for posture classification
        self.model = torch.load(checkpoint_path, map_location=self.device, weights_only=False)['model'].eval()

    def __del__(self):
        if hasattr(self, "blazepose") and self.blazepose is not None:
            self.blazepose.close()

    def first_stage(self, image_np: ndarray) -> list[list[float]]:
        """
        Detects and extracts keypoints from an image using BlazePose.
        Returns a list of keypoints (x, y, z, visibility) for each landmark.
        """
        keypoints = []
        result = self.blazepose.process(image_np)

        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return keypoints

    def second_stage(self, keypoints: ndarray) -> tuple[ndarray, ndarray]:
        """
        Passes the normalized keypoints through the neural network to classify the posture and correctness.
        Returns the probabilities for posture and correctness.
        """
        inputs = torch.tensor([keypoints], dtype=torch.float32).to(self.device)

        # Forward pass through the model
        with torch.no_grad():
            output: torch.Tensor = self.model(inputs)[0]

        # Split the output into posture and correctness logits
        posture_logits, correctness_logits = output.T

        # Apply softmax to posture logits and sigmoid to correctness logits
        posture_prob = F.softmax(posture_logits, dim=0)
        correctness_prob = torch.sigmoid(correctness_logits)

        return posture_prob.cpu().numpy(), correctness_prob.cpu().numpy()

    def process_image(self, image_np: ndarray, image_path: str = ""):
        """
        Full pipeline to process an image:
        1. Extract keypoints using BlazePose.
        2. Normalize the keypoints.
        3. Classify posture and correctness.
        4. Return structured results for analysis.
        """
        keypoints = self.first_stage(image_np)

        if not keypoints:
            print("No BlazePose landmarks detected in the image.")
            return {
                "image_path": image_path,
                "status": "no_landmarks",
            }

        # Normalize keypoints
        keypoints_np = np.array(keypoints)
        normalized_keypoints = self.normalize_keypoints(keypoints_np)

        # Classify posture and correctness
        posture_prob, correctness_prob = self.second_stage(normalized_keypoints)

        # Final prediction
        predicted_idx = int(posture_prob.argmax())
        predicted_posture = self.POSTURE_NAMES[predicted_idx]
        predicted_feedback = self.FEEDBACKS[round(float(correctness_prob[predicted_idx]))]

        # Top-2 posture info
        sorted_idx = np.argsort(posture_prob)[::-1]
        top1_idx = int(sorted_idx[0])
        top2_idx = int(sorted_idx[1])
        top1_prob = float(posture_prob[top1_idx])
        top2_prob = float(posture_prob[top2_idx])
        posture_margin = top1_prob - top2_prob

        print(f"Posture classification probabilities: {posture_prob.tolist()}")
        print(f"Correctness probabilities: {correctness_prob.tolist()}")
        print(f"Predicted posture index: {predicted_idx}")
        print(f"Predicted posture: {predicted_posture}")
        print(f"Predicted feedback: {predicted_feedback}")
        print(f"Corresponding correctness probability: {correctness_prob[predicted_idx]}")

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

    @staticmethod
    def normalize_keypoints(keypoints: ndarray) -> ndarray:
        """
        Normalizes the keypoints so that x, y are scaled to [0, 1] and z is normalized to a unit vector.
        """
        x, y, z, visibility = keypoints.T

        # Normalize x and y to the range [0, 1]
        scaler = MinMaxScaler()
        x[:] = scaler.fit_transform(x.reshape(-1, 1)).ravel()
        y[:] = scaler.fit_transform(y.reshape(-1, 1)).ravel()

        # Normalize z to a unit vector
        z /= np.linalg.norm(z)

        return keypoints


# %% Function to display an image from a file path
def show_image(image_path):
    # Load the image
    img = mpimg.imread(image_path)

    # Show the image
    plt.imshow(img)
    plt.title(image_path)
    plt.show()

# %% Function to parse true labels from image path
def parse_true_label_from_path(image_path: str):
    path_lower = image_path.lower().replace("\\", "/")

    if "downdog" in path_lower:
        true_posture = "Down Dog"
    elif "side_plank" in path_lower:
        true_posture = "Side Plank"
    elif "plank" in path_lower:
        true_posture = "Plank"
    elif "warrior_ii" in path_lower:
        true_posture = "Warrior II"
    else:
        true_posture = "Unknown"

    if "/positive/" in path_lower:
        true_feedback = "Correct"
    elif "/negative/" in path_lower:
        true_feedback = "Incorrect"
    else:
        true_feedback = "Unknown"

    return true_posture, true_feedback
# %% Load and process images
image_paths = [
    "./dataset/warrior_ii/positive/0326.png",
    "./dataset/plank/positive/frame_71.png",
    "./dataset/plank/positive/frame_293.png",
    "./dataset/downdog/negative/frame01255.png",
    "./dataset/warrior_ii/negative/frame_00185.png",
    "./dataset/plank/negative/frame_322.png",
    "./dataset/warrior_ii/negative/frame_00107.png",
    "./dataset/downdog/negative/frame_10149.png",
    "./dataset/side_plank/positive/frame_00100.png",
    "./dataset/side_plank/positive/frame_003.png",
    "./dataset/warrior_ii/positive/0249.png",
    "./dataset/warrior_ii/negative/frame_00238.png",
    "./dataset/warrior_ii/negative/frame_00191.png",
    "./dataset/side_plank/positive/frame_0032.png",
    "./dataset/plank/positive/frame_125.png",
    "./dataset/side_plank/positive/frame_00140.png",
    "dataset/warrior_ii/positive/0155.png",
    "dataset/warrior_ii/negative/frame_00256.png",
    "dataset/plank/positive/019.png",
    "dataset/plank/negative/ba_030.png",
]

# Load images into a list of NumPy arrays
image_list: list[ndarray] = []
for image_path in image_paths:
    image = Image.open(image_path)
    image_rgb = image.convert('RGB')
    image_np = np.array(image_rgb)
    image_list.append(image_np)

# %% Instantiate the Posture Correction System
posture_system = PostureCorrectionSystem()
print(flush=True)

# %% Process each image and display results
start_time = time()
rows = []

for image_path, image_np in zip(image_paths, image_list):
    print(f"Processing image: '{image_path}'")

    true_posture, true_feedback = parse_true_label_from_path(image_path)
    result: dict[str, object] = posture_system.process_image(image_np, image_path=image_path)

    if result is None:
        result = {
            "image_path": image_path,
            "status": "none"
        }

    result["true_posture"] = true_posture
    result["true_feedback"] = true_feedback

    if result.get("status") == "ok":
        result["is_posture_correct"] = int(result["pred_posture"] == true_posture)
        result["is_feedback_correct"] = int(result["pred_feedback"] == true_feedback)
        result["is_fully_correct"] = int(
            (result["pred_posture"] == true_posture) and
            (result["pred_feedback"] == true_feedback)
        )
    else:
        result["is_posture_correct"] = 0
        result["is_feedback_correct"] = 0
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

# %% Clean up resources
del posture_system
