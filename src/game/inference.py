"""
Real-time inference pipeline: webcam frame -> pose -> model predictions.

Maintains a rolling buffer of keypoints and runs both classifiers
on each new frame.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision

import config
from src.data.extract import ensure_model, extract_keypoints_from_result, normalize_keypoints
from src.data.preprocess import extract_head_features, flatten_window_for_punch
from src.models.punch_classifier import PunchClassifier
from src.models.defense_classifier import DefenseClassifier


# Standard MediaPipe 33-keypoint pose connections for skeleton drawing
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # face left
    (0, 4), (4, 5), (5, 6), (6, 8),        # face right
    (9, 10),                                 # mouth
    (11, 12),                                # shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm
    (11, 23), (12, 24), (23, 24),           # torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),  # right leg
]


class RealtimeInference:
    """
    Real-time pose estimation and action classification.

    Maintains a rolling buffer of normalized keypoints and runs
    both punch and defense classifiers on each new frame.
    """

    def __init__(self, punch_checkpoint: str | None = None,
                 defense_checkpoint: str | None = None,
                 device: torch.device | None = None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # MediaPipe Pose (Tasks API, 0.10+)
        model_path = ensure_model()
        base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_tasks_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_tasks_vision.RunningMode.VIDEO,
        )
        self.landmarker = mp_tasks_vision.PoseLandmarker.create_from_options(options)
        self._start_time = time.perf_counter()

        # Rolling buffer of raw keypoints (before normalization)
        self.keypoint_buffer = deque(maxlen=config.SEQUENCE_LENGTH)

        # Load models
        self.punch_model = self._load_model(
            PunchClassifier, punch_checkpoint or str(config.CHECKPOINTS_DIR / "punch_best.pt")
        )
        self.defense_model = self._load_model(
            DefenseClassifier, defense_checkpoint or str(config.CHECKPOINTS_DIR / "defense_best.pt")
        )

        # Temporal smoothing buffers
        self.punch_history = deque(maxlen=config.SMOOTHING_WINDOW)
        self.defense_history = deque(maxlen=config.SMOOTHING_WINDOW)

        # Performance tracking
        self.last_inference_time = 0.0

    def _load_model(self, model_class: type, checkpoint_path: str) -> nn.Module | None:
        """Load a model from checkpoint, or return None if not found."""
        try:
            model = model_class()
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            print(f"Loaded {model_class.__name__} from {checkpoint_path}")
            return model
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Warning: Could not load {model_class.__name__}: {e}")
            return None

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single BGR frame through the full pipeline.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            Dictionary with:
                - 'keypoints': raw (33, 3) array or None
                - 'punch_class': str or None
                - 'punch_confidence': float
                - 'defense_class': str or None
                - 'defense_confidence': float
                - 'pose_landmarks': list of NormalizedLandmark or None (for drawing)
                - 'inference_ms': inference time in milliseconds
        """
        t0 = time.perf_counter()

        result = {
            "keypoints": None,
            "punch_class": None,
            "punch_confidence": 0.0,
            "defense_class": None,
            "defense_confidence": 0.0,
            "pose_landmarks": None,
            "inference_ms": 0.0,
        }

        # Run MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.perf_counter() - self._start_time) * 1000)
        pose_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Store raw landmarks for drawing (first person only)
        if pose_result.pose_landmarks:
            result["pose_landmarks"] = pose_result.pose_landmarks[0]

        # Extract keypoints
        kp = extract_keypoints_from_result(pose_result)
        if kp is None:
            if self.keypoint_buffer:
                kp = self.keypoint_buffer[-1].copy()
            else:
                result["inference_ms"] = (time.perf_counter() - t0) * 1000
                return result

        result["keypoints"] = kp
        self.keypoint_buffer.append(kp)

        # Need full buffer for classification
        if len(self.keypoint_buffer) < config.SEQUENCE_LENGTH:
            result["inference_ms"] = (time.perf_counter() - t0) * 1000
            return result

        # Build sequence and normalize
        sequence = np.array(list(self.keypoint_buffer))  # (30, 33, 3)
        normalized = normalize_keypoints(sequence)

        # Punch classification
        if self.punch_model is not None:
            punch_input = flatten_window_for_punch(normalized)  # (30, 99)
            punch_tensor = torch.tensor(punch_input, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                punch_logits = self.punch_model(punch_tensor)
                punch_probs = torch.softmax(punch_logits, dim=1)[0]
                punch_idx = punch_probs.argmax().item()
                punch_conf = punch_probs[punch_idx].item()

            self.punch_history.append((punch_idx, punch_conf))
            smoothed_punch = self._smooth_predictions(self.punch_history)

            if smoothed_punch[1] >= config.CONFIDENCE_THRESHOLD:
                result["punch_class"] = config.PUNCH_CLASSES[smoothed_punch[0]]
                result["punch_confidence"] = smoothed_punch[1]

        # Defense classification
        if self.defense_model is not None:
            head_features = extract_head_features(normalized)  # (30, head_feat_dim)
            defense_tensor = torch.tensor(head_features, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                defense_logits = self.defense_model(defense_tensor)
                defense_probs = torch.softmax(defense_logits, dim=1)[0]
                defense_idx = defense_probs.argmax().item()
                defense_conf = defense_probs[defense_idx].item()

            self.defense_history.append((defense_idx, defense_conf))
            smoothed_defense = self._smooth_predictions(self.defense_history)

            if smoothed_defense[1] >= config.CONFIDENCE_THRESHOLD:
                result["defense_class"] = config.DEFENSE_CLASSES[smoothed_defense[0]]
                result["defense_confidence"] = smoothed_defense[1]

        result["inference_ms"] = (time.perf_counter() - t0) * 1000
        self.last_inference_time = result["inference_ms"]

        return result

    @staticmethod
    def _smooth_predictions(history: deque) -> tuple[int, float]:
        """
        Temporal smoothing via majority vote over recent predictions.

        Returns:
            (most_common_class, average_confidence_for_that_class)
        """
        if not history:
            return (0, 0.0)

        votes: dict[int, list[float]] = {}
        for cls, conf in history:
            votes.setdefault(cls, []).append(conf)

        best_cls = max(votes, key=lambda c: len(votes[c]))
        avg_conf = np.mean(votes[best_cls])

        return (best_cls, float(avg_conf))

    def draw_skeleton(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw pose skeleton on frame using OpenCV."""
        if landmarks is None:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]

        # Draw connections
        for start_idx, end_idx in _POSE_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                s = landmarks[start_idx]
                e = landmarks[end_idx]
                cv2.line(annotated,
                         (int(s.x * w), int(s.y * h)),
                         (int(e.x * w), int(e.y * h)),
                         (0, 0, 255), 2)

        # Draw keypoints
        for lm in landmarks:
            cv2.circle(annotated, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)

        return annotated

    def cleanup(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
