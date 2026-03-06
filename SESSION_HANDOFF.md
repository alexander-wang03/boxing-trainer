# Session Handoff -- Interactive Shadow Boxing Trainer

## Project Overview
Webcam-only interactive shadow boxing trainer using pose estimation + dual LSTM classifiers (punch + defense) with a Pygame-based real-time UI.

## Current Status: BoxingVI Dataset Integrated

Full scaffold complete (16 source files). BoxingVI punch dataset integrated and producing train/val/test splits. Defense model data still requires custom recording.

### Completed Files
- [x] `config.py` -- global constants, paths, hyperparameters, class labels (updated for COCO-17 keypoints)
- [x] `requirements.txt` -- all dependencies
- [x] `src/data/collect.py` -- webcam recording tool
- [x] `src/data/extract.py` -- MediaPipe keypoint extraction
- [x] `src/data/annotate.py` -- interactive annotation tool
- [x] `src/data/preprocess.py` -- normalization + augmentation (for custom data)
- [x] `src/data/dataset.py` -- PyTorch `BoxingDataset` + DataLoader factories
- [x] `src/data/load_boxingvi.py` -- **NEW** BoxingVI dataset loader + split builder
- [x] `src/models/punch_classifier.py` -- BiLSTM (updated: accepts configurable input_dim)
- [x] `src/models/defense_classifier.py` -- LSTM defense classifier
- [x] `src/models/baselines.py` -- Rule-based + Frame-SVM + Feedforward MLP
- [x] `src/training/train.py` -- training loop with early stopping, LR scheduling
- [x] `src/training/evaluate.py` -- evaluation + metrics
- [x] `src/game/inference.py` -- real-time prediction pipeline
- [x] `src/game/game_logic.py` -- sparring partner AI + scoring
- [x] `src/game/renderer.py` -- Pygame renderer
- [x] `src/game/app.py` -- main game loop

### BoxingVI Integration Details
- **Source:** https://github.com/Bikudebug/BoxingVI (Google Drive download)
- **Data placed at:** `data/boxingvi/{Annotation_files, Skeleton_data, RGB_videos}`
- **Format:** AlphaPose COCO-17 keypoints, 2D (x, y normalized), 25-frame clips
- **Classes mapped:** Jab->jab_left, Cross->cross_right, Lead Hook->hook_left, Rear Hook->hook_right, Lead Uppercut->uppercut_left, Rear Uppercut->uppercut_right
- **Split:** V1-V7 train (14,853 samples w/ augmentation), V8-V10 val+test (250+250)
- **Augmentation:** Horizontal flip (with L/R label swap) + speed variation (0.8-1.2x)
- **Output:** `data/splits/punch_{train,val,test}.npz` with shape `(N, 25, 34)`

### Config Changes (BoxingVI mode)
- `FEATURES_PER_FRAME` = 34 (was 99 for MediaPipe)
- `SEQUENCE_LENGTH` = 25 (was 30)
- `NUM_KEYPOINTS` = 17 (was 33)
- `KEYPOINT_DIMS` = 2 (was 3)
- Both MediaPipe and COCO constants preserved in config for future switching

### Next Steps
1. `pip install -r requirements.txt` (in venv)
2. `python -m src.data.load_boxingvi` -- already run, splits exist
3. `python -m src.training.train --model punch` -- train punch classifier on BoxingVI
4. `python -m src.training.evaluate --model punch` -- evaluate
5. Record custom defensive clips (slip, duck, weave, block) -- ~400 clips needed
6. Run real-time game: `python -m src.game.app`

### Issues & Fixes
1. **Unicode arrow in print statements (Windows cp1252):** `load_boxingvi.py` used Unicode arrows that failed on Windows console. Fixed by replacing with ASCII `->`.
2. **V6 different format:** V6 skeleton is raw per-frame `(46497, 1, 17, 3)` instead of pre-clipped `(N, 25, 17, 2)`. Handled with special-case `clip_v6_sequences()` function.
3. **Inconsistent annotation Excel layouts:** Each V1-V10 xlsx has different column arrangements and header rows. Handled by extracting first 3 non-NaN values per row (start, end, class) regardless of column structure.
4. **Class name case inconsistency:** BoxingVI has both "Lead Hook" and "Lead hook". Handled by lowercasing all class names before mapping.

### Architecture Decisions
- **Framework:** PyTorch for models, MediaPipe for real-time pose, Pygame for UI
- **Two-model approach:** Punch classifier uses full-body keypoints; Defense uses head keypoints + velocity
- **BoxingVI for punch data:** 6,915 clips (14,853 with augmentation) from public dataset, eliminating need for custom punch recording
- **Custom recording only for defense:** ~400 clips for slip/duck/weave/block still needed
- **Augmentation:** Horizontal flip (with L/R label swap), speed variation (0.8-1.2x)
- **Temporal smoothing:** Majority vote over last 5 predictions for stable real-time output

### Model Input Shapes (BoxingVI mode)
| Model | Params | Input |
|-------|--------|-------|
| Punch BiLSTM | ~500K | (25, 34) |
| Defense LSTM | ~200K | (25, head_features) |
| MLP Baseline | ~450K | (850,) flattened |
