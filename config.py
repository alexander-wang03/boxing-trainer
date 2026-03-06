"""
Global configuration for the Interactive Shadow Boxing Trainer.
Contains paths, hyperparameters, class labels, and constants.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
SPLITS_DIR = DATA_DIR / "splits"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

# ──────────────────────────────────────────────
# Class Labels
# ──────────────────────────────────────────────
PUNCH_CLASSES = [
    "neutral",
    "jab_left",
    "jab_right",
    "cross_left",
    "cross_right",
    "hook_left",
    "hook_right",
    "uppercut_left",
    "uppercut_right",
]
NUM_PUNCH_CLASSES = len(PUNCH_CLASSES)  # 9

DEFENSE_CLASSES = [
    "neutral",
    "slip",
    "duck",
    "weave",
    "block",
]
NUM_DEFENSE_CLASSES = len(DEFENSE_CLASSES)  # 5

# ──────────────────────────────────────────────
# MediaPipe / Keypoints
# ──────────────────────────────────────────────
NUM_KEYPOINTS = 33
KEYPOINT_DIMS = 3  # x, y, z
FEATURES_PER_FRAME = NUM_KEYPOINTS * KEYPOINT_DIMS  # 99

# Head keypoint indices (MediaPipe Pose)
HEAD_KEYPOINT_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # nose, eyes, ears, mouth

# ──────────────────────────────────────────────
# Sequence / Windowing
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 30  # frames (1 second at 30fps)
FRAME_RATE = 30

# ──────────────────────────────────────────────
# Camera Settings
# ──────────────────────────────────────────────
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
CAMERA_INDEX = 0

# ──────────────────────────────────────────────
# Model Hyperparameters
# ──────────────────────────────────────────────

# Punch Classifier (Model A)
PUNCH_FC_DIM = 128
PUNCH_LSTM_HIDDEN = 256
PUNCH_LSTM_LAYERS = 2
PUNCH_DROPOUT = 0.2

# Defense Classifier (Model B)
DEFENSE_LSTM_HIDDEN = 128
DEFENSE_LSTM_LAYERS = 2
DEFENSE_FC_DIM = 64
DEFENSE_DROPOUT = 0.2

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5

# Data split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Augmentation
SPEED_VARIATION_RANGE = (0.8, 1.2)
FRAME_DROP_PROB = 0.1

# ──────────────────────────────────────────────
# Inference / Real-time
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5  # frames for temporal smoothing

# ──────────────────────────────────────────────
# Game Settings
# ──────────────────────────────────────────────
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
ROUND_DURATION = 180  # seconds (3 minutes)
REST_DURATION = 60    # seconds (1 minute)
CUE_REACTION_WINDOW = 2.0  # seconds to respond to a defensive cue
CUE_MIN_INTERVAL = 2.0     # minimum seconds between cues
CUE_MAX_INTERVAL = 5.0     # maximum seconds between cues
