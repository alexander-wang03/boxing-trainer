# Data Collection

## Record all action classes (100 clips each, default timing)
python -m src.data.collect --auto

## Customize timing and clip count
python -m src.data.collect --auto --clips 50 --prep 1.5 --record 1.5 --rest 1.0

## Only record defensive moves
python -m src.data.collect --auto --actions slip duck weave block

The automated HUD guides you through each action:

GET READY (amber, 2.5s) — shows what action to perform
GO! (red, 1.0s) — perform the action now (recording)
REST (green, 0.5s) — relax before next clip
NEXT CLASS (purple, 3.0s) — pause before switching actions
Press ESC at any time to stop early.

### Punches

jab_left, jab_right, cross_left, cross_right, hook_left, hook_right, uppercut_left, uppercut_right

### Defense

slip, duck, weave, block

## Manual Mode

python -m src.data.collect --action duck

# Extract MediaPipe Keypoints

python -m src.data.extract

# Preprocess Data

python -m src.data.preprocess

# Train Data

python -m src.training.train --model punch
python -m src.training.train --model defense

# Train Data with baseline MLP

python -m src.training.train --model mlp_punch
python -m src.training.train --model mlp_defense

# Evaluate

python -m src.training.evaluate --model punch
python -m src.training.evaluate --model defense
python -m src.training.evaluate --compare