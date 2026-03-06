# Session Handoff — Interactive Shadow Boxing Trainer

## Project Overview
Webcam-only interactive shadow boxing trainer using MediaPipe pose estimation + dual LSTM classifiers (punch + defense) with a Pygame-based real-time UI.

## Current Status: Full Scaffold Complete

All 16 source files implemented. Project is ready for data collection and training.

### Completed Files
- [x] `config.py` — global constants, paths, hyperparameters, class labels
- [x] `requirements.txt` — all dependencies (PyTorch, MediaPipe, Pygame, OpenCV, etc.)
- [x] `src/data/collect.py` — webcam recording tool with keyboard controls (SPACE=record, S=save, Q=quit)
- [x] `src/data/extract.py` — MediaPipe keypoint extraction + shoulder-width normalization → .npy files
- [x] `src/data/annotate.py` — interactive annotation tool (play/pause, frame step, mark start/end → CSV)
- [x] `src/data/preprocess.py` — sliding windows, augmentation (flip, speed, frame drop), train/val/test split
- [x] `src/data/dataset.py` — PyTorch `BoxingDataset` + `get_punch_loaders()` / `get_defense_loaders()`
- [x] `src/models/punch_classifier.py` — BiLSTM: FC(99→128) → BiLSTM(256h, 2-layer) → FC(512→128→9), ~800K params
- [x] `src/models/defense_classifier.py` — LSTM: LSTM(66→128h, 2-layer) → FC(128→64→5), ~200K params
- [x] `src/models/baselines.py` — Rule-based + Frame-SVM + Feedforward MLP baselines
- [x] `src/training/train.py` — training loop with early stopping, LR scheduling, checkpointing
- [x] `src/training/evaluate.py` — accuracy, F1, confusion matrices, training curves, model comparison
- [x] `src/game/inference.py` — real-time pipeline: rolling buffer → normalize → dual model inference → temporal smoothing
- [x] `src/game/game_logic.py` — sparring partner AI, cue generation, combo scoring, difficulty levels
- [x] `src/game/renderer.py` — Pygame renderer: webcam feed, skeleton overlay, HUD, cues, flash effects, menus
- [x] `src/game/app.py` — main game loop tying together webcam + inference + game logic + rendering

### Next Steps (User Action Required)
1. **Install dependencies:** `pip install -r requirements.txt`
2. **Collect data:** `python -m src.data.collect --action jab_left` (repeat for each action type)
3. **Annotate clips:** `python -m src.data.annotate`
4. **Extract keypoints:** `python -m src.data.extract`
5. **Preprocess & split:** `python -m src.data.preprocess`
6. **Train models:** `python -m src.training.train --model punch` and `--model defense`
7. **Evaluate:** `python -m src.training.evaluate --compare`
8. **Run the game:** `python -m src.game.app`

### Issues & Fixes
_(none yet)_

### Architecture Decisions
- **Framework:** PyTorch for models, MediaPipe for pose, Pygame for UI
- **Two-model approach:** Punch classifier uses full-body keypoints (99 features/frame); Defense classifier uses head keypoints + velocity (66 features/frame)
- **Sequence length:** 30 frames (1 sec at 30fps) for both models
- **Normalization:** Shoulder-width normalization for translation/scale invariance
- **Augmentation:** Horizontal flip (with L/R label swap), speed variation (0.8–1.2x), random frame drop
- **Temporal smoothing:** Majority vote over last 5 predictions for stable real-time output
- **Game difficulty:** 3 levels controlling cue frequency and reaction window

### File Sizes (Approximate Parameters)
| Model | Params | Input |
|-------|--------|-------|
| Punch BiLSTM | ~800K | (30, 99) |
| Defense LSTM | ~200K | (30, 66) |
| MLP Baseline | ~1.6M | (2970,) flattened |
