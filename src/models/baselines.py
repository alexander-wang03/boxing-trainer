"""
Three baseline models for comparison.

Baseline 1: Rule-Based — geometric thresholds on keypoint positions
Baseline 2: Frame-SVM — hand-crafted features + RBF-kernel SVM
Baseline 3: Feedforward MLP — flattened sequences through an MLP
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config


# ──────────────────────────────────────────────
# Baseline 1: Rule-Based Classifier
# ──────────────────────────────────────────────

class RuleBasedClassifier:
    """
    Hand-coded geometric rules on keypoint positions.
    Uses the last frame vs first frame to detect motion patterns.
    """

    # MediaPipe indices
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def predict_punch(self, sequence: np.ndarray) -> int:
        """
        Classify punch type from a (seq_len, 33, 3) sequence.
        Returns index into PUNCH_CLASSES.
        """
        if len(sequence.shape) == 2:
            # (seq_len, 99) → (seq_len, 33, 3)
            sequence = sequence.reshape(-1, 33, 3)

        start = sequence[0]
        end = sequence[-1]

        # Wrist displacement
        lw_disp = end[self.LEFT_WRIST] - start[self.LEFT_WRIST]
        rw_disp = end[self.RIGHT_WRIST] - start[self.RIGHT_WRIST]

        lw_forward = -lw_disp[2]  # z decreases = forward
        rw_forward = -rw_disp[2]

        lw_up = -lw_disp[1]  # y decreases = up
        rw_up = -rw_disp[1]

        # Hip rotation (lateral displacement)
        hip_rotation = abs(end[self.LEFT_HIP, 0] - start[self.LEFT_HIP, 0])

        threshold_forward = 0.3
        threshold_up = 0.3
        threshold_rotation = 0.1

        # Detect which hand moved more
        left_active = abs(lw_forward) > abs(rw_forward)
        active_forward = lw_forward if left_active else rw_forward
        active_up = lw_up if left_active else rw_up
        side = "left" if left_active else "right"

        if abs(active_forward) < threshold_forward and abs(active_up) < threshold_up:
            return 0  # neutral

        if active_up > threshold_up and abs(active_forward) < threshold_forward:
            label = f"uppercut_{side}"
        elif hip_rotation > threshold_rotation and abs(active_forward) > threshold_forward:
            if active_forward > 0:
                label = f"cross_{side}"
            else:
                label = f"hook_{side}"
        elif active_forward > threshold_forward:
            label = f"jab_{side}"
        else:
            label = f"jab_{side}"

        return config.PUNCH_CLASSES.index(label) if label in config.PUNCH_CLASSES else 0

    def predict_defense(self, sequence: np.ndarray) -> int:
        """
        Classify defensive move from a (seq_len, 33, 3) sequence.
        Returns index into DEFENSE_CLASSES.
        """
        if len(sequence.shape) == 2:
            sequence = sequence.reshape(-1, 33, 3)

        start_nose = sequence[0, self.NOSE]
        end_nose = sequence[-1, self.NOSE]

        dy = end_nose[1] - start_nose[1]   # positive = down
        dx = end_nose[0] - start_nose[0]   # lateral movement

        duck_threshold = 0.15
        slip_threshold = 0.1
        weave_threshold = 0.08

        if dy > duck_threshold:
            return config.DEFENSE_CLASSES.index("duck")
        elif abs(dx) > slip_threshold and abs(dy) < duck_threshold:
            return config.DEFENSE_CLASSES.index("slip")
        elif abs(dx) > weave_threshold and dy > weave_threshold:
            return config.DEFENSE_CLASSES.index("weave")

        # Block: check if wrists are near head
        end_lw = sequence[-1, self.LEFT_WRIST]
        end_rw = sequence[-1, self.RIGHT_WRIST]
        end_nose_pos = sequence[-1, self.NOSE]
        lw_dist = np.linalg.norm(end_lw[:2] - end_nose_pos[:2])
        rw_dist = np.linalg.norm(end_rw[:2] - end_nose_pos[:2])

        if lw_dist < 0.15 and rw_dist < 0.15:
            return config.DEFENSE_CLASSES.index("block")

        return 0  # neutral

    def predict_batch(self, sequences: np.ndarray, task: str = "punch") -> np.ndarray:
        """Predict labels for a batch of sequences."""
        fn = self.predict_punch if task == "punch" else self.predict_defense
        return np.array([fn(seq) for seq in sequences])


# ──────────────────────────────────────────────
# Baseline 2: Frame-Level SVM
# ──────────────────────────────────────────────

class FrameSVM:
    """
    Hand-crafted features per frame aggregated over window, then SVM.
    """

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
        ])

    @staticmethod
    def extract_features(sequence: np.ndarray) -> np.ndarray:
        """
        Extract hand-crafted features from a sequence.

        Args:
            sequence: (seq_len, 99) or (seq_len, 33, 3)

        Returns:
            1D feature vector.
        """
        if len(sequence.shape) == 2 and sequence.shape[1] == 99:
            sequence = sequence.reshape(-1, 33, 3)

        seq_len = len(sequence)

        # Joint angles (elbow angles)
        features = []

        for frame in sequence:
            # Left elbow angle
            v1 = frame[11] - frame[13]  # shoulder to elbow
            v2 = frame[15] - frame[13]  # wrist to elbow
            cos_left = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

            # Right elbow angle
            v1 = frame[12] - frame[14]
            v2 = frame[16] - frame[14]
            cos_right = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

            # Wrist positions relative to shoulder
            lw_rel = frame[15] - frame[11]
            rw_rel = frame[16] - frame[12]

            # Nose position
            nose = frame[0]

            features.append(np.concatenate([
                [cos_left, cos_right],
                lw_rel, rw_rel,
                nose,
            ]))

        features = np.array(features)

        # Aggregate: mean, std, min, max over time
        agg = np.concatenate([
            features.mean(axis=0),
            features.std(axis=0),
            features.min(axis=0),
            features.max(axis=0),
        ])

        # Add velocity features
        velocity = np.diff(features, axis=0)
        if len(velocity) > 0:
            agg = np.concatenate([
                agg,
                velocity.mean(axis=0),
                velocity.std(axis=0),
            ])

        return agg

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit SVM on feature-extracted sequences."""
        features = np.array([self.extract_features(seq) for seq in X])
        self.pipeline.fit(features, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for sequences."""
        features = np.array([self.extract_features(seq) for seq in X])
        return self.pipeline.predict(features)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)


# ──────────────────────────────────────────────
# Baseline 3: Feedforward MLP
# ──────────────────────────────────────────────

class FeedforwardMLP(nn.Module):
    """
    Flatten 30-frame sequence → 2-layer MLP.
    No temporal modeling.
    """

    def __init__(
        self,
        input_dim: int = config.SEQUENCE_LENGTH * config.FEATURES_PER_FRAME,
        hidden1: int = 512,
        hidden2: int = 256,
        num_classes: int = config.NUM_PUNCH_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.flatten_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) or (batch, flat_dim)

        Returns:
            logits: (batch, num_classes)
        """
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        return self.net(x)


if __name__ == "__main__":
    # Test all baselines
    print("=== Baseline 1: Rule-Based ===")
    rb = RuleBasedClassifier()
    dummy_seq = np.random.randn(30, 33, 3).astype(np.float32)
    print(f"  Punch prediction: {config.PUNCH_CLASSES[rb.predict_punch(dummy_seq)]}")
    print(f"  Defense prediction: {config.DEFENSE_CLASSES[rb.predict_defense(dummy_seq)]}")

    print("\n=== Baseline 2: Frame-SVM ===")
    svm = FrameSVM()
    feat = svm.extract_features(dummy_seq)
    print(f"  Feature vector length: {len(feat)}")

    print("\n=== Baseline 3: Feedforward MLP ===")
    mlp = FeedforwardMLP()
    params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")
    dummy_tensor = torch.randn(4, config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME)
    out = mlp(dummy_tensor)
    print(f"  Input: {dummy_tensor.shape}, Output: {out.shape}")
