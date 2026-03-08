# Session Handoff -- Interactive Shadow Boxing Trainer

## Project Overview
Webcam-only interactive shadow boxing trainer using pose estimation + dual LSTM classifiers (punch + defense) with a Pygame-based real-time UI.

## Current Status: Custom Data Collection Required

Full scaffold complete (17 source files). BoxingVI dataset was integrated but **proved insufficient for punch classification** -- custom data collection is needed for all classes (punches + defense). Automated collection tool with game-style HUD is ready.

### Completed Files
- [x] `config.py` -- global constants, paths, hyperparameters, class labels (COCO-17 keypoint mode)
- [x] `requirements.txt` -- all dependencies
- [x] `src/data/collect.py` -- webcam recording tool (manual + automated game-style mode)
- [x] `src/data/extract.py` -- MediaPipe keypoint extraction
- [x] `src/data/annotate.py` -- interactive annotation tool
- [x] `src/data/preprocess.py` -- normalization + augmentation (for custom data)
- [x] `src/data/dataset.py` -- PyTorch `BoxingDataset` + DataLoader factories
- [x] `src/data/load_boxingvi.py` -- BoxingVI dataset loader (kept for reference, not primary data source)
- [x] `src/models/punch_classifier.py` -- BiLSTM with packed-sequence padding handling
- [x] `src/models/defense_classifier.py` -- LSTM defense classifier
- [x] `src/models/baselines.py` -- Rule-based + Frame-SVM + Feedforward MLP
- [x] `src/training/train.py` -- training loop with early stopping, LR scheduling
- [x] `src/training/evaluate.py` -- evaluation + metrics
- [x] `src/game/inference.py` -- real-time prediction pipeline
- [x] `src/game/game_logic.py` -- sparring partner AI + scoring
- [x] `src/game/renderer.py` -- Pygame renderer
- [x] `src/game/app.py` -- main game loop

### Key Decision: BoxingVI Insufficient
BoxingVI dataset (6,915 AlphaPose COCO-17 skeleton clips from YouTube boxing videos) was integrated and tested but is **not good enough** for our punch classification needs. All training data must be collected custom using the automated collection tool.

### Data Collection Plan
**Total clips needed:** ~1,200 (100 per class)

Punches (8 classes x 100 each = 800):
- `jab_left`, `jab_right`, `cross_left`, `cross_right`
- `hook_left`, `hook_right`, `uppercut_left`, `uppercut_right`

Defense (4 classes x 100 each = 400):
- `slip`, `duck`, `weave`, `block`

**Collection command:**
```
python -m src.data.collect --auto --clips 100
```

**Important: Auto mode does NOT track progress across sessions.** If you stop at 50 clips and restart, specify `--clips 50` for the remaining. Files won't be overwritten (timestamped filenames), but the counter resets to 0.

**Resuming example:**
```
# Day 1: started with 100, stopped at 60 for jab_left
# Day 2: record remaining 40
python -m src.data.collect --auto --clips 40 --actions jab_left
```

### User Modifications (since initial scaffold)
- **`collect.py`**: Added automated collection mode (`--auto`) with game-style HUD (GET READY/GO!/REST/NEXT CLASS phases, progress bar, counters)
- **`punch_classifier.py`**: Added `pack_padded_sequence` handling to skip zero-padded frames in BiLSTM
- **`load_boxingvi.py`**: Added `normalize_to_body()` (hip-center + shoulder-width normalization) and `add_velocity_features()` (frame-to-frame velocity appended to positions)
- **`config.py`**: Tuned hyperparams (FC_DIM 128->64, LSTM_HIDDEN 256->128, dropout 0.2->0.4, patience 10->20), added `COCO_FEATURES_PER_FRAME_WITH_VEL = 68`

### Next Steps
1. **Collect custom data:** `python -m src.data.collect --auto --clips 100`
2. **Extract keypoints:** `python -m src.data.extract`
3. **Annotate clips:** `python -m src.data.annotate`
4. **Preprocess & split:** `python -m src.data.preprocess`
5. **Train models:** `python -m src.training.train --model punch` and `--model defense`
6. **Evaluate:** `python -m src.training.evaluate --compare`
7. **Run the game:** `python -m src.game.app`

### Issues & Fixes
1. **Unicode arrow in print statements (Windows cp1252):** `load_boxingvi.py` used Unicode arrows that failed on Windows console. Fixed by replacing with ASCII `->`.
2. **V6 different format:** V6 skeleton is raw per-frame `(46497, 1, 17, 3)` instead of pre-clipped `(N, 25, 17, 2)`. Handled with special-case `clip_v6_sequences()` function.
3. **Inconsistent annotation Excel layouts:** Each V1-V10 xlsx has different column arrangements and header rows. Handled by extracting first 3 non-NaN values per row.
4. **Class name case inconsistency:** BoxingVI has both "Lead Hook" and "Lead hook". Handled by lowercasing all class names before mapping.
5. **BoxingVI data quality:** Insufficient for punch classification -- decided to collect custom data for all classes.
6. **Auto collection no resume:** Counter resets on restart. User must manually specify remaining clip count.

### Architecture Decisions
- **Framework:** PyTorch for models, MediaPipe for real-time pose, Pygame for UI
- **Two-model approach:** Punch classifier uses full-body keypoints; Defense uses head keypoints + velocity
- **Custom data for everything:** BoxingVI abandoned; all data collected via automated tool
- **Augmentation:** Horizontal flip (with L/R label swap), speed variation (0.8-1.2x)
- **Packed sequences:** BiLSTM uses `pack_padded_sequence` to handle zero-padded clips correctly
- **Body normalization:** Hip-center translation + shoulder-width scaling
- **Temporal smoothing:** Majority vote over last 5 predictions for stable real-time output

### Current Config (Hyperparameters)
| Parameter | Value |
|-----------|-------|
| FEATURES_PER_FRAME | 34 (COCO-17) |
| SEQUENCE_LENGTH | 25 |
| PUNCH_FC_DIM | 64 |
| PUNCH_LSTM_HIDDEN | 128 |
| PUNCH_DROPOUT | 0.4 |
| BATCH_SIZE | 32 |
| LEARNING_RATE | 0.001 |
| EARLY_STOPPING_PATIENCE | 20 |
| LR_SCHEDULER_PATIENCE | 8 |
