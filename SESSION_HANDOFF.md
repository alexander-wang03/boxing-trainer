# Session Handoff -- Interactive Shadow Boxing Trainer

## Project Overview
Webcam-only interactive shadow boxing trainer using pose estimation + dual LSTM classifiers (punch + defense) with a Pygame-based real-time UI.

## Current Status: Extraction Running -- Ready for Preprocessing & Training

Full scaffold complete (17 source files). Custom data has been collected and keypoint extraction is currently running (or complete). Next: preprocess splits, train both models.

### Completed Files
- [x] `config.py` -- global constants, paths, hyperparameters, class labels (MediaPipe mode)
- [x] `requirements.txt` -- all dependencies
- [x] `src/data/collect.py` -- webcam recording tool (manual + automated game-style mode)
- [x] `src/data/extract.py` -- MediaPipe keypoint extraction (updated for 0.10+ Tasks API)
- [x] `src/data/annotate.py` -- interactive annotation tool
- [x] `src/data/preprocess.py` -- normalization + augmentation (for custom data)
- [x] `src/data/dataset.py` -- PyTorch `BoxingDataset` + DataLoader factories
- [x] `src/data/load_boxingvi.py` -- BoxingVI dataset loader (kept for reference, not primary data source)
- [x] `src/models/punch_classifier.py` -- BiLSTM with packed-sequence padding handling
- [x] `src/models/defense_classifier.py` -- LSTM defense classifier
- [x] `src/models/baselines.py` -- Rule-based + Frame-SVM + Feedforward MLP
- [x] `src/training/train.py` -- training loop with early stopping, LR scheduling
- [x] `src/training/evaluate.py` -- evaluation + metrics
- [x] `src/game/inference.py` -- real-time prediction pipeline (updated for 0.10+ Tasks API)
- [x] `src/game/game_logic.py` -- sparring partner AI + scoring
- [x] `src/game/renderer.py` -- Pygame renderer
- [x] `src/game/app.py` -- main game loop

### Key Decision: BoxingVI Insufficient
BoxingVI dataset (6,915 AlphaPose COCO-17 skeleton clips from YouTube boxing videos) was integrated and tested but is **not good enough** for our punch classification needs. All training data must be collected custom using the automated collection tool.

### Data Collection -- COMPLETE
**Total clips collected:** 1,215 across 12 classes

| Class | Count |
|-------|-------|
| jab_left | 100 |
| jab_right | 100 |
| cross_left | 100 |
| cross_right | 100 |
| hook_left | 100 |
| hook_right | 115 |
| uppercut_left | 100 |
| uppercut_right | 100 |
| slip | 100 |
| duck | 100 |
| weave | 100 |
| block | 100 |

### Next Steps
1. ~~**Fix config mismatch**~~ -- DONE (switched to MediaPipe mode: 33 kpts, 3D, 99 features, seq_len=30)
2. ~~**Extract keypoints**~~ -- IN PROGRESS (`python -m src.data.extract`, runs MediaPipe on raw .mp4 -> .npy in data/processed/)
3. **Preprocess & split:** `python -m src.data.preprocess` (sliding windows, augmentation, train/val/test .npz)
4. **Train models:** `python -m src.training.train --model punch` and `--model defense`
5. **Evaluate:** `python -m src.training.evaluate --compare`
6. **Run the game:** `python -m src.game.app`

### Active Config (MediaPipe / Custom Data Mode)
| Parameter | Value |
|-----------|-------|
| NUM_KEYPOINTS | 33 (MediaPipe) |
| KEYPOINT_DIMS | 3 (x, y, z) |
| FEATURES_PER_FRAME | 99 |
| SEQUENCE_LENGTH | 30 |
| HEAD_KEYPOINT_INDICES | [0..10] (MediaPipe) |
| PUNCH_FC_DIM | 64 |
| PUNCH_LSTM_HIDDEN | 128 |
| PUNCH_DROPOUT | 0.4 |
| BATCH_SIZE | 32 |
| LEARNING_RATE | 0.001 |
| EARLY_STOPPING_PATIENCE | 20 |
| LR_SCHEDULER_PATIENCE | 8 |

### Issues & Fixes
1. **Unicode arrow in print statements (Windows cp1252):** `load_boxingvi.py` used Unicode arrows that failed on Windows console. Fixed by replacing with ASCII `->`.
2. **V6 different format:** V6 skeleton is raw per-frame `(46497, 1, 17, 3)` instead of pre-clipped `(N, 25, 17, 2)`. Handled with special-case `clip_v6_sequences()` function.
3. **Inconsistent annotation Excel layouts:** Each V1-V10 xlsx has different column arrangements and header rows. Handled by extracting first 3 non-NaN values per row.
4. **Class name case inconsistency:** BoxingVI has both "Lead Hook" and "Lead hook". Handled by lowercasing all class names before mapping.
5. **BoxingVI data quality:** Insufficient for punch classification -- decided to collect custom data for all classes.
6. **Auto collection no resume:** Counter resets on restart. User must manually specify remaining clip count.
7. **MediaPipe 0.10+ API change:** `mp.solutions.pose` removed. Fixed by migrating `extract.py` and `inference.py` to the new `mp.tasks.vision.PoseLandmarker` Tasks API. Requires `pose_landmarker_full.task` model file (auto-downloaded to project root on first run, ~25MB). Skeleton drawing now done manually via OpenCV instead of `mp.solutions.drawing_utils`.

### Architecture Decisions
- **Framework:** PyTorch for models, MediaPipe for real-time pose, Pygame for UI
- **Two-model approach:** Punch classifier uses full-body keypoints (99 feat); Defense uses head keypoints + velocity
- **Custom data for everything:** BoxingVI abandoned; all data collected via automated tool
- **Augmentation:** Horizontal flip (with L/R label swap), speed variation (0.8-1.2x)
- **Packed sequences:** BiLSTM uses `pack_padded_sequence` to handle zero-padded clips correctly
- **Body normalization:** Hip-center translation + shoulder-width scaling
- **Temporal smoothing:** Majority vote over last 5 predictions for stable real-time output
- **MediaPipe model file:** `pose_landmarker_full.task` in project root -- do not delete
