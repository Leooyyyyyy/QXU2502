from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from core.stage3_utils import normalize_keypoints

DEFAULT_CHECKPOINT_PATH = Path("artifacts/experiments/phase1.2_baseline_refined/best_checkpoint.pth")
DEFAULT_CONFIG_PATH = Path("artifacts/experiments/phase1.2_baseline_refined/config.json")
DEFAULT_FEATURE_MODE = "landmarks_v1"


def get_device() -> torch.device:
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


@dataclass
class RuntimeConfig:
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)
    config_path: str = str(DEFAULT_CONFIG_PATH)
    feature_mode: str = DEFAULT_FEATURE_MODE
    confidence_threshold: float = 0.55
    posture_margin_threshold: float = 0.08
    correctness_threshold: float = 0.50
    min_visible_landmarks: int = 18
    min_landmark_visibility: float = 0.35
    model_complexity: int = 1
    camera_index: int = 0
    stabilization_window_size: int = 5
    stabilization_min_votes: int = 3
    posture_window_size: int = 4
    posture_min_votes: int = 2
    correctness_window_size: int = 5
    correctness_min_votes: int = 3
    subtype_window_size: int = 7
    subtype_min_votes: int = 4
    subtype_confidence_threshold: float = 0.60
    subtype_margin_threshold: float = 0.10
    static_image_mode: bool = True


@dataclass
class FrozenModelBundle:
    checkpoint_path: str
    config_path: str | None
    experiment_name: str
    model_name: str
    input_shape: list[int]
    feature_mode: str
    metadata: dict[str, Any]
    config: dict[str, Any]
    device: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["posture_names"] = self.metadata["posture_names"]
        payload["posture_dirs"] = self.metadata["posture_dirs"]
        payload["negative_subtypes_by_posture"] = self.metadata["negative_subtypes_by_posture"]
        payload["subtype_counts_by_posture"] = self.metadata["subtype_counts_by_posture"]
        return payload


class TemporalStabilizer:
    def __init__(self, window_size: int = 5, min_votes: int = 3):
        self.window_size = max(1, window_size)
        self.min_votes = max(1, min_votes)
        self.history: deque[str] = deque(maxlen=self.window_size)
        self.payloads: deque[dict[str, Any]] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.history.clear()
        self.payloads.clear()

    def update(self, result: dict[str, Any]) -> dict[str, Any]:
        if result["status"] in {"no_person", "out_of_frame"}:
            self.reset()
            result["stability"] = 0.0
            result["stability_votes"] = 0
            return result
        if result["status"] == "low_confidence":
            result["stability"] = len(self.history) / self.window_size if self.history else 0.0
            result["stability_votes"] = 0
            return result

        signature = self._signature(result)
        self.history.append(signature)
        self.payloads.append(result)
        counts = Counter(self.history)
        winning_signature, votes = counts.most_common(1)[0]
        stability = votes / len(self.history)

        result["stability"] = stability
        result["stability_votes"] = votes

        if votes < self.min_votes:
            return result

        stable_result = next(payload for payload in reversed(self.payloads) if self._signature(payload) == winning_signature)
        merged = dict(stable_result)
        merged["raw_result"] = result
        merged["stability"] = stability
        merged["stability_votes"] = votes
        merged["status_message"] = f"Stable ({votes}/{len(self.history)} recent frames)"
        return merged

    @staticmethod
    def _signature(result: dict[str, Any]) -> str:
        subtype = result.get("pred_negative_subtype") or "-"
        return f"{result['pred_posture']}|{result['pred_feedback']}|{subtype}"


class LabelStabilizer:
    def __init__(self, window_size: int, min_votes: int):
        self.window_size = max(1, window_size)
        self.min_votes = max(1, min_votes)
        self.history: deque[str] = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self.history.clear()

    def update(self, label: str | None) -> dict[str, Any]:
        if label is None:
            self.reset()
            return {
                "stable_label": None,
                "stability": 0.0,
                "votes": 0,
                "history_size": 0,
                "is_stable": False,
            }

        self.history.append(label)
        counts = Counter(self.history)
        winning_label, votes = counts.most_common(1)[0]
        history_size = len(self.history)
        stability = votes / history_size
        is_stable = votes >= self.min_votes
        return {
            "stable_label": winning_label if is_stable else None,
            "stability": stability,
            "votes": votes,
            "history_size": history_size,
            "is_stable": is_stable,
        }


class HierarchicalStabilizer:
    def __init__(self, runtime_config: RuntimeConfig):
        self.posture = LabelStabilizer(runtime_config.posture_window_size, runtime_config.posture_min_votes)
        self.correctness = LabelStabilizer(
            runtime_config.correctness_window_size,
            runtime_config.correctness_min_votes,
        )
        self.subtype = LabelStabilizer(runtime_config.subtype_window_size, runtime_config.subtype_min_votes)

    def reset(self) -> None:
        self.posture.reset()
        self.correctness.reset()
        self.subtype.reset()

    def update(self, result: dict[str, Any]) -> dict[str, Any]:
        status = result["status"]
        if status in {"no_person", "out_of_frame"}:
            self.reset()
            result["stability"] = 0.0
            result["stability_votes"] = 0
            result["stable_posture"] = None
            result["stable_feedback"] = None
            result["stable_subtype"] = None
            result["subtype_gated"] = True
            result["subtype_gate_reason"] = "pose_unavailable"
            result["display_negative_subtype"] = None
            result["display_label"] = result.get("pred_posture", "-")
            result["feedback_text"] = "Move into frame"
            return result

        posture_state = self.posture.update(result.get("pred_posture"))
        correctness_input = result.get("pred_feedback") if posture_state["is_stable"] else None
        correctness_state = self.correctness.update(correctness_input)

        subtype_input = None
        if correctness_state["is_stable"] and correctness_state["stable_label"] == "Incorrect":
            subtype_input = result.get("pred_negative_subtype")
        subtype_state = self.subtype.update(subtype_input)

        result["stable_posture"] = posture_state["stable_label"]
        result["stable_feedback"] = correctness_state["stable_label"]
        result["stable_subtype"] = subtype_state["stable_label"]
        result["posture_stability"] = posture_state["stability"]
        result["posture_stability_votes"] = posture_state["votes"]
        result["correctness_stability"] = correctness_state["stability"]
        result["correctness_stability_votes"] = correctness_state["votes"]
        result["subtype_stability"] = subtype_state["stability"]
        result["subtype_stability_votes"] = subtype_state["votes"]
        result["stability"] = min(posture_state["stability"], correctness_state["stability"]) if correctness_state["history_size"] else posture_state["stability"]
        result["stability_votes"] = min(posture_state["votes"], correctness_state["votes"]) if correctness_state["history_size"] else posture_state["votes"]

        gated_subtype, gate_reason = self._gate_subtype(result, posture_state, correctness_state, subtype_state)
        result["display_negative_subtype"] = gated_subtype
        result["subtype_gated"] = gated_subtype is None
        result["subtype_gate_reason"] = gate_reason

        display_posture = posture_state["stable_label"] or result["pred_posture"]
        display_feedback = correctness_state["stable_label"] or result["pred_feedback"]
        display_label = f"{display_posture} | {display_feedback}"
        feedback_text = "Hold the current position" if display_feedback == "Correct" else "Incorrect posture detected"
        if gated_subtype is not None:
            display_label = f"{display_label} | {gated_subtype}"
            feedback_text = f"Correction focus: {gated_subtype.replace('_', ' ')}"

        result["display_label"] = display_label
        result["feedback_text"] = feedback_text

        status_message_parts = []
        if posture_state["is_stable"]:
            status_message_parts.append(f"posture {posture_state['votes']}/{posture_state['history_size']}")
        if correctness_state["is_stable"]:
            status_message_parts.append(
                f"feedback {correctness_state['votes']}/{correctness_state['history_size']}"
            )
        if gated_subtype is not None:
            status_message_parts.append(f"subtype {subtype_state['votes']}/{subtype_state['history_size']}")
        elif gate_reason != "not_incorrect":
            status_message_parts.append(f"subtype gated: {gate_reason.replace('_', ' ')}")
        if status_message_parts:
            result["status_message"] = "Stable: " + ", ".join(status_message_parts)

        return result

    @staticmethod
    def _gate_subtype(
        result: dict[str, Any],
        posture_state: dict[str, Any],
        correctness_state: dict[str, Any],
        subtype_state: dict[str, Any],
    ) -> tuple[str | None, str]:
        if result["status"] != "ok":
            return None, "status_not_ok"
        if not posture_state["is_stable"]:
            return None, "posture_not_stable"
        if not correctness_state["is_stable"]:
            return None, "correctness_not_stable"
        if correctness_state["stable_label"] != "Incorrect":
            return None, "not_incorrect"
        if result.get("pred_negative_subtype") is None:
            return None, "no_subtype_candidate"
        if result.get("pred_negative_subtype_prob") is None:
            return None, "no_subtype_probability"
        if result["pred_negative_subtype_prob"] < result["subtype_confidence_threshold"]:
            return None, "subtype_confidence_low"
        if result["pred_negative_subtype_margin"] < result["subtype_margin_threshold"]:
            return None, "subtype_margin_low"
        if not subtype_state["is_stable"]:
            return None, "subtype_not_stable"
        return subtype_state["stable_label"], "accepted"


class Phase12InferenceEngine:
    def __init__(self, runtime_config: RuntimeConfig | None = None):
        self.runtime_config = runtime_config or RuntimeConfig()
        self.device = get_device()
        checkpoint = torch.load(self.runtime_config.checkpoint_path, map_location=self.device, weights_only=False)
        self.bundle = self._load_bundle(checkpoint)
        self.model = self._load_model(checkpoint)
        self.metadata = self.bundle.metadata
        self.posture_names = self.metadata["posture_names"]
        self.posture_dirs = self.metadata["posture_dirs"]
        self.negative_subtypes_by_posture = self.metadata["negative_subtypes_by_posture"]
        self.subtype_counts_by_posture = self.metadata["subtype_counts_by_posture"]
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=self.runtime_config.static_image_mode,
            model_complexity=self.runtime_config.model_complexity,
        )
        self.stabilizer = TemporalStabilizer(
            window_size=self.runtime_config.stabilization_window_size,
            min_votes=self.runtime_config.stabilization_min_votes,
        )
        self.hierarchical_stabilizer = HierarchicalStabilizer(self.runtime_config)

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()
            self.pose = None

    def __del__(self) -> None:
        self.close()

    def describe_frozen_model(self) -> dict[str, Any]:
        description = self.bundle.to_dict()
        description["runtime_target"] = "Report-aligned Phase 1.2: Baseline After Warrior II Refinement"
        description["runtime_target_path"] = str(DEFAULT_CHECKPOINT_PATH.parent)
        description["metadata_note"] = (
            "This runtime target is aligned to report Phase 1.2. "
            "Some internal checkpoint or config names may still use stale baseline_refined labels."
        )
        description["preprocessing"] = {
            "x": "per-frame min-max normalization",
            "y": "per-frame min-max normalization",
            "z": "per-frame L2 normalization",
            "visibility": "passed into feature tensor and zero-masked by model forward",
        }
        return description

    def predict_image_file(self, image_path: str, use_stabilization: bool = False) -> dict[str, Any]:
        image_np = np.array(Image.open(image_path).convert("RGB"))
        result = self.predict_rgb_frame(image_np, use_stabilization=use_stabilization)
        result["image_path"] = image_path
        return result

    def predict_rgb_frame(
        self,
        image_rgb: np.ndarray,
        use_stabilization: bool = False,
        use_hierarchical_stabilization: bool = False,
    ) -> dict[str, Any]:
        keypoints = self._extract_keypoints(image_rgb)
        if keypoints is None:
            result = {
                "status": "no_person",
                "status_message": "No pose landmarks detected",
            }
            if use_hierarchical_stabilization:
                return self.hierarchical_stabilizer.update(result)
            return self.stabilizer.update(result) if use_stabilization else result

        visibility_summary = self._summarize_visibility(keypoints)
        if visibility_summary["visible_landmarks"] < self.runtime_config.min_visible_landmarks:
            result = {
                "status": "out_of_frame",
                "status_message": "Pose is incomplete or partially out of frame",
                "visible_landmarks": visibility_summary["visible_landmarks"],
                "mean_visibility": visibility_summary["mean_visibility"],
            }
            if use_hierarchical_stabilization:
                return self.hierarchical_stabilizer.update(result)
            return self.stabilizer.update(result) if use_stabilization else result

        features = self._build_features(keypoints)
        outputs = self._forward(features)
        result = self._decode_outputs(outputs, visibility_summary)

        if use_hierarchical_stabilization:
            result = self._attach_subtype_gate_thresholds(result)
            result = self.hierarchical_stabilizer.update(result)
        elif use_stabilization:
            result = self.stabilizer.update(result)
        return result

    def _attach_subtype_gate_thresholds(self, result: dict[str, Any]) -> dict[str, Any]:
        result["subtype_confidence_threshold"] = self.runtime_config.subtype_confidence_threshold
        result["subtype_margin_threshold"] = self.runtime_config.subtype_margin_threshold
        return result

    def _load_bundle(self, checkpoint: dict[str, Any]) -> FrozenModelBundle:
        checkpoint_path = Path(self.runtime_config.checkpoint_path)
        config_path = Path(self.runtime_config.config_path)
        metadata = checkpoint["metadata"]
        experiment_config = checkpoint.get("config", {})
        if not experiment_config and config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                experiment_config = json.load(f)

        return FrozenModelBundle(
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path) if config_path.exists() else None,
            experiment_name=checkpoint.get("experiment_name", "phase1.1_baseline_refined"),
            model_name=type(checkpoint["model"]).__name__,
            input_shape=[33, 4],
            feature_mode=self.runtime_config.feature_mode,
            metadata=metadata,
            config=experiment_config,
            device=str(self.device),
        )

    def _load_model(self, checkpoint: dict[str, Any]) -> torch.nn.Module:
        model = checkpoint["model"].to(self.device)
        model.eval()
        return model

    def _extract_keypoints(self, image_rgb: np.ndarray) -> np.ndarray | None:
        result = self.pose.process(image_rgb)
        if not result.pose_landmarks:
            return None

        keypoints = []
        for landmark in result.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        return np.array(keypoints, dtype=np.float32)

    def _summarize_visibility(self, keypoints: np.ndarray) -> dict[str, float]:
        visibility = keypoints[:, 3]
        visible_landmarks = int(np.sum(visibility >= self.runtime_config.min_landmark_visibility))
        mean_visibility = float(np.mean(visibility))
        return {
            "visible_landmarks": visible_landmarks,
            "mean_visibility": mean_visibility,
        }

    def _build_features(self, keypoints: np.ndarray) -> np.ndarray:
        if self.runtime_config.feature_mode != DEFAULT_FEATURE_MODE:
            raise ValueError(
                f"Unsupported feature_mode '{self.runtime_config.feature_mode}'. "
                f"Current runtime supports '{DEFAULT_FEATURE_MODE}' only."
            )

        normalized = np.array(keypoints, copy=True)
        normalize_keypoints(normalized)
        return normalized

    def _forward(self, features: np.ndarray) -> dict[str, torch.Tensor]:
        inputs = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(inputs)

    def _decode_outputs(self, outputs: dict[str, torch.Tensor], visibility_summary: dict[str, float]) -> dict[str, Any]:
        posture_prob = F.softmax(outputs["posture_logits"][0], dim=0).cpu().numpy()
        correctness_prob = torch.sigmoid(outputs["correctness_logits"][0]).cpu().numpy()
        negative_logits = outputs["negative_subtype_logits"][0]

        sorted_posture_indices = np.argsort(posture_prob)[::-1]
        predicted_posture_idx = int(sorted_posture_indices[0])
        runner_up_idx = int(sorted_posture_indices[1])

        posture_confidence = float(posture_prob[predicted_posture_idx])
        posture_margin = float(posture_prob[predicted_posture_idx] - posture_prob[runner_up_idx])
        selected_correctness_prob = float(correctness_prob[predicted_posture_idx])
        predicted_feedback = (
            "Correct" if selected_correctness_prob >= self.runtime_config.correctness_threshold else "Incorrect"
        )

        predicted_negative_subtype = None
        predicted_negative_subtype_prob = None
        predicted_negative_subtype_margin = 0.0
        negative_subtype_distribution = None
        if predicted_feedback == "Incorrect":
            posture_dir = self.posture_dirs[predicted_posture_idx]
            valid_count = self.subtype_counts_by_posture[predicted_posture_idx]
            negative_subtype_distribution = F.softmax(negative_logits[predicted_posture_idx, :valid_count], dim=0).cpu().numpy()
            negative_subtype_idx = int(np.argmax(negative_subtype_distribution))
            predicted_negative_subtype = self.negative_subtypes_by_posture[posture_dir][negative_subtype_idx]
            predicted_negative_subtype_prob = float(negative_subtype_distribution[negative_subtype_idx])
            sorted_subtype_prob = np.sort(negative_subtype_distribution)[::-1]
            if len(sorted_subtype_prob) > 1:
                predicted_negative_subtype_margin = float(sorted_subtype_prob[0] - sorted_subtype_prob[1])

        status = "ok"
        status_message = "Prediction accepted"
        if posture_confidence < self.runtime_config.confidence_threshold:
            status = "low_confidence"
            status_message = "Posture confidence below threshold"
        elif posture_margin < self.runtime_config.posture_margin_threshold:
            status = "low_confidence"
            status_message = "Top posture predictions are too close"

        display_label = f"{self.posture_names[predicted_posture_idx]} | {predicted_feedback}"
        feedback_text = "Hold the current position"
        if predicted_feedback == "Incorrect" and predicted_negative_subtype is not None:
            display_label = f"{display_label} | {predicted_negative_subtype}"
            feedback_text = f"Correction focus: {predicted_negative_subtype.replace('_', ' ')}"

        result = {
            "status": status,
            "status_message": status_message,
            "pred_posture_idx": predicted_posture_idx,
            "pred_posture": self.posture_names[predicted_posture_idx],
            "pred_feedback": predicted_feedback,
            "pred_negative_subtype": predicted_negative_subtype,
            "posture_confidence": posture_confidence,
            "posture_margin": posture_margin,
            "selected_correctness_prob": selected_correctness_prob,
            "pred_negative_subtype_prob": predicted_negative_subtype_prob,
            "pred_negative_subtype_margin": predicted_negative_subtype_margin,
            "display_negative_subtype": predicted_negative_subtype,
            "subtype_gated": False,
            "subtype_gate_reason": "baseline_display",
            "display_label": display_label,
            "feedback_text": feedback_text,
            "top2_posture": self.posture_names[runner_up_idx],
            "top2_posture_prob": float(posture_prob[runner_up_idx]),
            "visible_landmarks": visibility_summary["visible_landmarks"],
            "mean_visibility": visibility_summary["mean_visibility"],
            "posture_probabilities": {
                posture_name: float(posture_prob[idx]) for idx, posture_name in enumerate(self.posture_names)
            },
            "correctness_probabilities": {
                posture_name: float(correctness_prob[idx]) for idx, posture_name in enumerate(self.posture_names)
            },
        }

        if negative_subtype_distribution is not None:
            posture_dir = self.posture_dirs[predicted_posture_idx]
            result["negative_subtype_probabilities"] = {
                subtype_name: float(prob)
                for subtype_name, prob in zip(
                    self.negative_subtypes_by_posture[posture_dir],
                    negative_subtype_distribution,
                )
            }

        return result
