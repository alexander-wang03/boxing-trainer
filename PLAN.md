# Implementation Plan: Interactive Shadow Boxing Trainer

## Tech Stack
- **Python 3.10+**, **PyTorch** (models/training), **MediaPipe** (pose estimation)
- **Pygame** (real-time UI/game window), **OpenCV** (webcam capture)
- **NumPy**, **Pandas** (data processing), **scikit-learn** (baselines, metrics)

---

## Project Structure

```
boxing-trainer/
├── requirements.txt
├── README.md
├── config.py                    # Global constants (paths, hyperparams, class labels)
│
├── data/
│   ├── raw/                     # Raw recorded video clips (not committed)
│   ├── processed/               # Extracted keypoint sequences (.npy files)
│   ├── annotations/             # CSV annotation files (frame ranges, labels)
│   └── splits/                  # train/val/test split CSVs
│
├── src/
│   ├── data/
│   │   ├── collect.py           # Webcam recording tool for data collection
│   │   ├── annotate.py          # Annotation helper (review clips, label actions)
│   │   ├── extract.py           # MediaPipe keypoint extraction from videos
│   │   ├── preprocess.py        # Normalize, window, augment keypoint sequences
│   │   └── dataset.py           # PyTorch Dataset/DataLoader classes
│   │
│   ├── models/
│   │   ├── baselines.py         # Rule-based, SVM, and MLP baselines
│   │   ├── punch_classifier.py  # Model A: BiLSTM punch classifier
│   │   └── defense_classifier.py# Model B: LSTM defense classifier
│   │
│   ├── training/
│   │   ├── train.py             # Training loop (shared for both models)
│   │   └── evaluate.py          # Evaluation: accuracy, F1, confusion matrices
│   │
│   └── game/
│       ├── app.py               # Main Pygame application loop
│       ├── renderer.py          # Drawing: webcam feed, overlays, HUD, cues
│       ├── game_logic.py        # Sparring partner AI, cue generation, scoring
│       └── inference.py         # Real-time pose → model prediction pipeline
│
├── checkpoints/                 # Saved model weights (.pt files)
├── notebooks/                   # Jupyter notebooks for exploration/visualization
│   └── exploration.ipynb
├── results/                     # Training logs, plots, confusion matrices
└── tests/                       # Unit tests
    └── test_data_pipeline.py
```

---

## Phase 1: Data Collection & Recording Tool (`src/data/collect.py`)

- OpenCV webcam capture at 30fps (Logitech C920, 1920x1080)
- Pygame or CV2 preview window showing live feed
- Keyboard controls: press key to start/stop recording a clip
- Each clip saved as individual video file in `data/raw/` with metadata (timestamp, action type)
- Record ~1,200 action clips total:
  - 800 punches: jab/cross/hook/uppercut × left/right (100 each)
  - 400 defense: slip/duck/weave/block (100 each)

## Phase 2: Annotation & Keypoint Extraction

### `src/data/annotate.py`
- Playback recorded clips, mark start/end frame of each action
- Save annotations to CSV: `filename, action_type, hand, start_frame, end_frame`

### `src/data/extract.py`
- Run MediaPipe Pose on each video frame
- Extract 33 keypoints × 3 coordinates (x, y, z) per frame
- Normalize keypoints to shoulder width (translation + scale invariant)
- Save as `.npy` arrays in `data/processed/`

## Phase 3: Data Preprocessing & PyTorch Dataset (`src/data/preprocess.py`, `src/data/dataset.py`)

### Preprocessing
- Sliding window: 30-frame sequences (1 second at 30fps)
- Body-relative normalization: translate to hip-midpoint origin, scale by shoulder width (makes predictions translation- and scale-invariant)
- Augmentation:
  - Horizontal flip (mirror left/right labels) — for body-relative coords, negate x-axis (`-x`), **not** `1 - x`
  - Speed variation: interpolate to 0.8x–1.2x speed
  - Random frame dropping (simulate occlusion)
- Note: velocity features (frame-to-frame differences) may hurt performance if the pose estimator is noisy (e.g. AlphaPose); omit for noisy inputs and enable only with smooth per-frame estimates (e.g. MediaPipe)
- Split: 70% train / 15% val / 15% test, stratified by class

### PyTorch Dataset
- `PunchDataset`: returns (sequence_tensor [30, 99], label) for punch model
- `DefenseDataset`: returns (sequence_tensor [30, head_features], label) for defense model
- DataLoaders with batch_size=32, shuffle, num_workers

## Phase 4: Baseline Models (`src/models/baselines.py`)

> **Format consistency note:** all three baselines must use the same keypoint format and feature dimensionality as the LSTM they are benchmarked against. If training on custom MediaPipe data (33 kp × 3D = 99 features), baselines must also consume 99-feature sequences. Rule-based and SVM baselines contain hardcoded MediaPipe joint indices and will not work correctly with other formats (e.g. AlphaPose COCO-17).

### Baseline 1: Rule-Based
- Hand-coded geometric rules on keypoint positions (MediaPipe indices):
  - Jab: wrist extension beyond shoulder + same-side elbow angle increase
  - Cross: wrist extension + hip rotation (cross-body)
  - Defense: head displacement thresholds (down=duck, lateral=slip, etc.)
- Returns predicted class based on threshold checks

### Baseline 2: Frame-Level SVM
- Extract hand-crafted features per frame: joint angles, distances, velocities
- Aggregate over window (mean, max, std)
- Train RBF-kernel SVM via scikit-learn

### Baseline 3: Feedforward MLP
- Flatten 30-frame sequence → 2970D input (30 × 99 for MediaPipe)
- Architecture: 2970 → 512 → 256 → 9 (punch) or → 5 (defense)
- ReLU activations, dropout

## Phase 5: LSTM Models

### Model A: Punch Classifier (`src/models/punch_classifier.py`)
- Input: [batch, 30, 99] (30 frames × 33 keypoints × 3 coords)
- Per-frame FC layer: 99 → 64, ReLU, Dropout(0.4)
- 2-layer Bidirectional LSTM: input_size=64, hidden_size=128
- Classifier head: 256 → 64 → 9 (8 punch types + neutral)
- ~600K parameters
- **Critical implementation note:** use `pack_padded_sequence` / `pad_packed_sequence` to handle variable-length sequences (zero-padded clips). Reading `lstm_out[:, -1, :]` on padded sequences causes the backward LSTM pass to consume zeros, collapsing all outputs to near-identical values and stalling accuracy at chance level. Gather the output at the last *valid* frame index instead.

### Model B: Defense Classifier (`src/models/defense_classifier.py`)
- Input: [batch, 30, head_features] (head keypoints + optional velocity features)
- Head feature extraction: nose, ears, eyes positions (MediaPipe subset)
- 2-layer LSTM: input_size=head_features, hidden_size=128
- Classifier head: 128 → 64 → 5 (slip, duck, weave, block, neutral)
- ~200K parameters

### Training (`src/training/train.py`)
- Loss: CrossEntropyLoss (with class weights if imbalanced)
- Optimizer: Adam, lr=0.001, weight_decay=1e-3 (L2 regularization)
- LR scheduler: ReduceLROnPlateau (patience=8, factor=0.5)
- Early stopping on validation loss (patience=20)
- Fixed random seed (seed=42) for reproducibility
- Save best checkpoint to `checkpoints/`
- Log training/val loss and accuracy per epoch

### Evaluation (`src/training/evaluate.py`)
- Per-class accuracy, precision, recall, F1
- Confusion matrix visualization (saved to `results/`)
- Compare all models (baselines vs LSTMs) in summary table

## Phase 6: Real-Time Inference Pipeline (`src/game/inference.py`)

> **Train/inference format dependency:** the model loaded here must have been trained on **MediaPipe Pose** data (33 keypoints × 3D = 99 features, `SEQUENCE_LENGTH=30`). A model trained on the BoxingVI dataset (AlphaPose COCO-17, 17 kp × 2D = 34 features) cannot be used directly in the game — the feature dimensions differ. Custom recording (`src/data/collect.py`) is required before the game is functional end-to-end.

- Initialize MediaPipe Pose + load trained model checkpoints
- Maintain rolling buffer of last `SEQUENCE_LENGTH` (30) frames of keypoints
- On each new frame:
  1. Extract keypoints via MediaPipe (33 kp × 3D)
  2. Apply body-relative normalization (same pipeline as training: center on hip midpoint, scale by shoulder width)
  3. Push to rolling buffer
  4. Run punch model + defense model inference (batch=1)
  5. Apply confidence threshold + temporal smoothing
- `SEQUENCE_LENGTH` and `FEATURES_PER_FRAME` in `config.py` must match the values used during training
- Target: >20 fps, <100ms latency
- Return: `(punch_class, punch_conf, defense_class, defense_conf)`

## Phase 7: Game Logic & Sparring Partner (`src/game/game_logic.py`)

- **Sparring partner AI**: generates attack cues at intervals
  - Cue types: "SLIP LEFT", "DUCK", "BLOCK HIGH", "WEAVE RIGHT"
  - Difficulty scaling: adjustable cue frequency and reaction window
- **Scoring system**:
  - Award points for correct punches and successful defenses
  - Track combo streaks
  - Deduct for missed defensive cues
- **Session management**: rounds with rest periods, stats tracking

## Phase 8: Pygame UI (`src/game/app.py`, `src/game/renderer.py`)

### `app.py` — Main Loop
- Initialize Pygame window + webcam capture
- Game states: MENU → TRAINING → ROUND_END → RESULTS
- Main loop: capture frame → inference → game logic → render → display
- Handle keyboard/mouse input for menu navigation

### `renderer.py` — Drawing
- Render webcam feed as background (via OpenCV → Pygame surface)
- Overlay skeleton visualization (draw keypoints + connections)
- HUD elements: score, combo counter, round timer
- Attack cue display: directional prompts with countdown timer
- Visual feedback: green flash on correct response, red on miss
- Results screen: accuracy breakdown, session stats

---

## Dataset Strategy

### BoxingVI (pre-existing dataset)
- Used for **academic comparison only**: train LSTM punch classifier and MLP baseline, report cross-subject test accuracy
- Format: AlphaPose COCO-17 keypoints (17 kp × 2D = 34 features), clips of 10–25 frames, zero-padded to 25
- Available: V1–V10 (10 subjects); V1–V7 train, V8–V10 test (cross-subject generalization)
- 6 punch classes available (neutral class absent; jab_right / cross_left synthesized via flip augmentation)
- Expected accuracy ceiling ~60–65% due to cross-subject domain shift
- **Cannot be used to power the real-time game** (different keypoint format than MediaPipe)

### Custom MediaPipe Data (required for the game)
- Record ~1,200 clips using `src/data/collect.py` with Logitech C920 at 30fps:
  - 800 punches: jab/cross/hook/uppercut × left/right (100 each)
  - 400 defense: slip/duck/weave/block (100 each)
- Extract 33 MediaPipe keypoints × 3D = 99 features, 30-frame windows
- This dataset is what trains the deployed game models; target: >75% punch, >70% defense accuracy

---

## File Implementation Order

1. `config.py` — constants, paths, class labels
2. `requirements.txt` — all dependencies
3. `src/data/collect.py` — recording tool (needed first for data)
4. `src/data/extract.py` — keypoint extraction
5. `src/data/annotate.py` — annotation helper
6. `src/data/preprocess.py` — normalization + augmentation
7. `src/data/dataset.py` — PyTorch datasets
8. `src/models/punch_classifier.py` — Model A
9. `src/models/defense_classifier.py` — Model B
10. `src/models/baselines.py` — all three baselines
11. `src/training/train.py` — training loop
12. `src/training/evaluate.py` — evaluation + metrics
13. `src/game/inference.py` — real-time prediction pipeline
14. `src/game/game_logic.py` — sparring partner + scoring
15. `src/game/renderer.py` — Pygame drawing
16. `src/game/app.py` — main application entry point

---

## Success Criteria
- Punch classification accuracy: >75%
- Defense classification accuracy: >70%
- Real-time performance: >20 fps, <100ms inference latency
- LSTM models outperform all three baselines
