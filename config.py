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
# Keypoint Configuration
# ──────────────────────────────────────────────

# MediaPipe Pose (used for real-time inference + custom data)
MEDIAPIPE_NUM_KEYPOINTS = 33
MEDIAPIPE_KEYPOINT_DIMS = 3  # x, y, z
MEDIAPIPE_FEATURES_PER_FRAME = MEDIAPIPE_NUM_KEYPOINTS * MEDIAPIPE_KEYPOINT_DIMS  # 99

# AlphaPose / COCO-17 (used by BoxingVI dataset)
COCO_NUM_KEYPOINTS = 17
COCO_KEYPOINT_DIMS = 2  # x, y (normalized)
COCO_FEATURES_PER_FRAME = COCO_NUM_KEYPOINTS * COCO_KEYPOINT_DIMS  # 34

# Active keypoint config — set based on which dataset is being used.
# Change these when switching between BoxingVI (COCO) and custom (MediaPipe).
NUM_KEYPOINTS = COCO_NUM_KEYPOINTS
KEYPOINT_DIMS = COCO_KEYPOINT_DIMS
FEATURES_PER_FRAME = COCO_FEATURES_PER_FRAME  # 34

# Head keypoint indices
HEAD_KEYPOINT_INDICES_MEDIAPIPE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # nose, eyes, ears, mouth
HEAD_KEYPOINT_INDICES_COCO = [0, 1, 2, 3, 4]  # nose, left eye, right eye, left ear, right ear
HEAD_KEYPOINT_INDICES = HEAD_KEYPOINT_INDICES_COCO

# ──────────────────────────────────────────────
# BoxingVI Dataset
# ──────────────────────────────────────────────
BOXINGVI_DIR = DATA_DIR / "boxingvi"
BOXINGVI_SEQUENCE_LENGTH = 25  # BoxingVI clips are 25 frames (zero-padded)

# ──────────────────────────────────────────────
# Sequence / Windowing
# ──────────────────────────────────────────────
SEQUENCE_LENGTH = 25  # frames (matches BoxingVI; change to 30 for custom 30fps data)
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
