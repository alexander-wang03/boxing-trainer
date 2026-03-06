"""
BoxingVI dataset integration loader.

Parses the BoxingVI dataset (10 videos, AlphaPose COCO-17 keypoints)
and converts it into train/val/test .npz splits compatible with our pipeline.

Dataset structure expected at data/boxingvi/:
    Annotation_files/  — V1.xlsx ... V10.xlsx (inconsistent column layouts)
    Skeleton_data/     — V1.npy ... V10.npy  (pre-extracted AlphaPose keypoints)

Usage:
    python -m src.data.load_boxingvi
    python -m src.data.load_boxingvi --output_dir data/splits --no-augment
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config

# ──────────────────────────────────────────────
# BoxingVI-specific constants
# ──────────────────────────────────────────────

BOXINGVI_DIR = config.DATA_DIR / "boxingvi"
BOXINGVI_ANNOTATION_DIR = BOXINGVI_DIR / "Annotation_files"
BOXINGVI_SKELETON_DIR = BOXINGVI_DIR / "Skeleton_data"

# BoxingVI uses AlphaPose COCO-17 keypoints (x, y normalized)
COCO_NUM_KEYPOINTS = 17
COCO_KEYPOINT_DIMS = 2  # x, y
COCO_FEATURES_PER_FRAME = COCO_NUM_KEYPOINTS * COCO_KEYPOINT_DIMS  # 34

BOXINGVI_SEQUENCE_LENGTH = 25  # All clips zero-padded to 25 frames

# Map BoxingVI class names (with case normalization) to our punch classes.
# BoxingVI uses "Lead"/"Rear" where Lead = left hand, Rear = right hand
# (orthodox stance convention).
BOXINGVI_CLASS_MAP = {
    "jab": "jab_left",
    "cross": "cross_right",
    "lead hook": "hook_left",
    "rear hook": "hook_right",
    "lead uppercut": "uppercut_left",
    "rear uppercut": "uppercut_right",
}

# Videos assigned to train vs val split (following paper: S1-S15 train, S16-S20 val)
# The paper maps subjects to videos but doesn't specify exactly which.
# We use V1-V7 as train, V8-V10 as val/test (roughly 80/20).
TRAIN_VIDEOS = ["V1", "V2", "V3", "V4", "V5", "V6", "V7"]
VAL_TEST_VIDEOS = ["V8", "V9", "V10"]


def parse_annotations(video_id: str) -> list[dict]:
    """
    Parse a single BoxingVI annotation Excel file.

    Handles the inconsistent column layouts across V1-V10 by extracting
    the first 3 meaningful columns: start_frame, end_frame, class_name.

    Returns:
        List of dicts with keys: 'start', 'end', 'class_name'
    """
    xlsx_path = BOXINGVI_ANNOTATION_DIR / f"{video_id}.xlsx"
    df = pd.read_excel(xlsx_path, header=None)

    annotations = []

    for _, row in df.iterrows():
        # Get first 3 non-NaN values from the row
        vals = [v for v in row.values if pd.notna(v)]
        if len(vals) < 3:
            continue

        # Try to identify the pattern: (number, number, string)
        start_val, end_val, class_val = vals[0], vals[1], vals[2]

        # Skip header rows
        class_str = str(class_val).strip()
        if class_str.lower() in ("class", "class ", "type", "nan", ""):
            continue

        # Skip rows where start/end aren't numeric
        try:
            start = int(float(start_val))
            end = int(float(end_val))
        except (ValueError, TypeError):
            continue

        # Normalize class name
        class_normalized = class_str.strip().lower()

        if class_normalized not in BOXINGVI_CLASS_MAP:
            continue

        annotations.append({
            "start": start,
            "end": end,
            "class_name": class_normalized,
        })

    return annotations


def load_skeleton_data(video_id: str) -> np.ndarray | None:
    """
    Load skeleton .npy for a video. Returns array of shape (N, 25, 17, 2).

    Handles V6's special format: (46497, 1, 17, 3) -> needs clipping via annotations.
    All other videos are already pre-clipped: (N, 25, 17, 2).
    """
    npy_path = BOXINGVI_SKELETON_DIR / f"{video_id}.npy"
    data = np.load(npy_path, allow_pickle=True)

    if video_id == "V6":
        # V6 is raw per-frame data (46497, 1, 17, 3) — x, y, confidence
        # We only use x, y (first 2 channels)
        # Clipping into sequences is done in load_boxingvi_dataset()
        return data
    else:
        # Already pre-clipped: (N, 25, 17, 2)
        if data.shape[1:] == (25, 17, 2):
            return data
        else:
            print(f"  Warning: {video_id} has unexpected shape {data.shape}, skipping.")
            return None


def clip_v6_sequences(raw_data: np.ndarray, annotations: list[dict]) -> np.ndarray:
    """
    Extract 25-frame clips from V6's raw per-frame skeleton data.

    V6 raw shape: (total_frames, 1, 17, 3) — we take x, y only.
    For each annotation, extract frames [start:start+25] and zero-pad if needed.

    Returns:
        Array of shape (num_clips, 25, 17, 2)
    """
    total_frames = raw_data.shape[0]
    clips = []

    for ann in annotations:
        start = ann["start"]
        end = min(ann["end"] + 1, total_frames)
        length = end - start

        if start >= total_frames or length <= 0:
            # Zero clip
            clips.append(np.zeros((BOXINGVI_SEQUENCE_LENGTH, COCO_NUM_KEYPOINTS, COCO_KEYPOINT_DIMS)))
            continue

        # Extract frames, take x/y only (drop confidence)
        frames = raw_data[start:end, 0, :, :2]  # (length, 17, 2)

        # Pad or truncate to 25 frames
        if len(frames) < BOXINGVI_SEQUENCE_LENGTH:
            pad = np.zeros((BOXINGVI_SEQUENCE_LENGTH - len(frames), COCO_NUM_KEYPOINTS, COCO_KEYPOINT_DIMS))
            frames = np.concatenate([frames, pad], axis=0)
        else:
            frames = frames[:BOXINGVI_SEQUENCE_LENGTH]

        clips.append(frames)

    return np.array(clips, dtype=np.float32)


def normalize_to_body(window: np.ndarray) -> np.ndarray:
    """
    Normalize COCO-17 keypoints to be body-relative.

    Centers on hip midpoint (kp 11, 12) and scales by shoulder width (kp 5, 6).
    Uses only non-zero (non-padded) frames for computing normalization parameters.
    Zero-padded frames remain zero after normalization.

    Args:
        window: Shape (seq_len, 17, 2).

    Returns:
        Normalized array of same shape.
    """
    # Detect valid (non-padded) frames — zero-padded frames have all keypoints at (0, 0)
    frame_norms = np.linalg.norm(window.reshape(window.shape[0], -1), axis=1)
    valid_mask = frame_norms > 1e-6

    if not valid_mask.any():
        return window

    valid = window[valid_mask]  # (n_valid, 17, 2)

    # Compute mean hip center from valid frames only
    hip_center = ((valid[:, 11, :] + valid[:, 12, :]) / 2.0).mean(axis=0)  # (2,)

    # Compute mean shoulder width from valid frames only
    shoulder_width = np.linalg.norm(valid[:, 5, :] - valid[:, 6, :], axis=1)
    scale = shoulder_width.mean()

    if scale < 1e-6:
        return window

    normalized = window.copy()
    # Apply normalization only to valid frames; padded frames stay as zeros
    normalized[valid_mask] = (valid - hip_center[np.newaxis, np.newaxis, :]) / scale
    return normalized


def add_velocity_features(clip: np.ndarray) -> np.ndarray:
    """
    Append frame-to-frame velocity to position features.

    Args:
        clip: Shape (seq_len, 17, 2) — body-normalized positions.

    Returns:
        Shape (seq_len, 17, 4) — positions + velocities concatenated along last axis.
        Velocity is zeroed at padded frames and at the frame after the last valid frame.
    """
    seq_len = clip.shape[0]

    # Detect valid frames
    frame_norms = np.linalg.norm(clip.reshape(seq_len, -1), axis=1)
    valid_mask = frame_norms > 1e-6

    # Compute velocity: diff of consecutive frames, prepend zero for frame 0
    vel = np.zeros_like(clip)  # (seq_len, 17, 2)
    vel[1:] = clip[1:] - clip[:-1]

    # Zero out velocity on padded frames and at the transition boundary
    for t in range(seq_len):
        if not valid_mask[t]:
            vel[t] = 0.0
        elif t > 0 and not valid_mask[t - 1]:
            vel[t] = 0.0  # first valid frame after padding has no valid predecessor

    return np.concatenate([clip, vel], axis=-1)  # (seq_len, 17, 4)


def augment_horizontal_flip_coco(window: np.ndarray) -> np.ndarray:
    """
    Mirror COCO-17 keypoints horizontally (flip x-coordinates + swap L/R pairs).

    Args:
        window: Shape (seq_len, 17, 2).

    Returns:
        Flipped array of same shape.
    """
    flipped = window.copy()
    flipped[:, :, 0] = -flipped[:, :, 0]  # body-relative: negate x to mirror

    # COCO L/R swap pairs
    swap_pairs = [
        (1, 2),    # eyes
        (3, 4),    # ears
        (5, 6),    # shoulders
        (7, 8),    # elbows
        (9, 10),   # wrists
        (11, 12),  # hips
        (13, 14),  # knees
        (15, 16),  # ankles
    ]
    for l, r in swap_pairs:
        flipped[:, [l, r]] = flipped[:, [r, l]]

    return flipped


def augment_speed_variation_clip(window: np.ndarray,
                                 speed_range: tuple = config.SPEED_VARIATION_RANGE) -> np.ndarray:
    """Resample a (seq_len, 17, 2) clip at random speed."""
    seq_len = window.shape[0]
    speed = np.random.uniform(*speed_range)
    new_len = max(2, int(seq_len * speed))

    src_indices = np.linspace(0, seq_len - 1, new_len)
    tgt_indices = np.linspace(0, seq_len - 1, seq_len)

    result = np.zeros_like(window)
    for kp in range(window.shape[1]):
        for dim in range(window.shape[2]):
            resampled = np.interp(src_indices, np.arange(seq_len), window[:, kp, dim])
            result[:, kp, dim] = np.interp(tgt_indices, src_indices, resampled)
    return result


def get_flipped_label(class_name: str) -> str:
    """Swap left/right in a mapped label after horizontal flip."""
    if "left" in class_name:
        return class_name.replace("left", "right")
    elif "right" in class_name:
        return class_name.replace("right", "left")
    return class_name


def load_boxingvi_dataset(augment: bool = True, verbose: bool = True):
    """
    Load the entire BoxingVI dataset and return sequences + labels.

    Returns:
        train_X: np.ndarray of shape (N_train, 25, 34)
        train_y: np.ndarray of shape (N_train,)
        valtest_X: np.ndarray of shape (N_valtest, 25, 34)
        valtest_y: np.ndarray of shape (N_valtest,)
    """
    all_data = {"train": ([], []), "valtest": ([], [])}

    for video_id in [f"V{i}" for i in range(1, 11)]:
        split = "train" if video_id in TRAIN_VIDEOS else "valtest"

        # Parse annotations
        annotations = parse_annotations(video_id)
        if not annotations:
            if verbose:
                print(f"  {video_id}: no valid annotations, skipping.")
            continue

        # Load skeleton data
        skeleton = load_skeleton_data(video_id)
        if skeleton is None:
            continue

        # Handle V6 specially
        if video_id == "V6":
            clips = clip_v6_sequences(skeleton, annotations)
        else:
            # Pre-clipped data — verify count matches
            num_clips = skeleton.shape[0]
            num_anns = len(annotations)

            if num_clips != num_anns:
                # Use whichever is smaller
                n = min(num_clips, num_anns)
                if verbose:
                    print(f"  {video_id}: clip/annotation mismatch "
                          f"({num_clips} vs {num_anns}), using {n}")
                clips = skeleton[:n]
                annotations = annotations[:n]
            else:
                clips = skeleton

        sequences_list, labels_list = all_data[split]

        for idx, ann in enumerate(annotations):
            if idx >= len(clips):
                break

            clip = clips[idx]  # (25, 17, 2)
            clip = normalize_to_body(clip)
            mapped_label = BOXINGVI_CLASS_MAP[ann["class_name"]]
            label_idx = config.PUNCH_CLASSES.index(mapped_label)

            # Flatten to (25, 34) for model input
            flat = clip.reshape(BOXINGVI_SEQUENCE_LENGTH, -1).astype(np.float32)
            sequences_list.append(flat)
            labels_list.append(label_idx)

            if augment and split == "train":
                # Horizontal flip (clip is already normalized, flip after)
                flipped_clip = augment_horizontal_flip_coco(clip)
                flipped_label = get_flipped_label(mapped_label)
                if flipped_label in config.PUNCH_CLASSES:
                    flat_flip = flipped_clip.reshape(BOXINGVI_SEQUENCE_LENGTH, -1).astype(np.float32)
                    sequences_list.append(flat_flip)
                    labels_list.append(config.PUNCH_CLASSES.index(flipped_label))

                # Speed variation
                speed_clip = augment_speed_variation_clip(clip)
                flat_speed = speed_clip.reshape(BOXINGVI_SEQUENCE_LENGTH, -1).astype(np.float32)
                sequences_list.append(flat_speed)
                labels_list.append(label_idx)

        if verbose:
            print(f"  {video_id}: {len(annotations)} annotations -> "
                  f"{len(sequences_list)} samples ({split})")

    train_X = np.array(all_data["train"][0], dtype=np.float32)
    train_y = np.array(all_data["train"][1], dtype=np.int64)
    valtest_X = np.array(all_data["valtest"][0], dtype=np.float32)
    valtest_y = np.array(all_data["valtest"][1], dtype=np.int64)

    if verbose:
        print(f"\nTotal train: {len(train_X)}, val+test: {len(valtest_X)}")

    return train_X, train_y, valtest_X, valtest_y


def build_splits(output_dir: Path = config.SPLITS_DIR,
                 augment: bool = True):
    """
    Build train/val/test .npz splits from BoxingVI data.

    Saves:
        punch_train.npz, punch_val.npz, punch_test.npz
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading BoxingVI dataset...")
    train_X, train_y, valtest_X, valtest_y = load_boxingvi_dataset(augment=augment)

    if len(train_X) == 0:
        print("Error: No training data loaded.")
        return

    # Split val+test into val and test (50/50)
    if len(valtest_X) > 0:
        val_X, test_X, val_y, test_y = train_test_split(
            valtest_X, valtest_y, test_size=0.5,
            stratify=valtest_y, random_state=42
        )
    else:
        # Fall back to splitting from train
        train_X, temp_X, train_y, temp_y = train_test_split(
            train_X, train_y, test_size=0.3,
            stratify=train_y, random_state=42
        )
        val_X, test_X, val_y, test_y = train_test_split(
            temp_X, temp_y, test_size=0.5,
            stratify=temp_y, random_state=42
        )

    print(f"\nFinal splits:")
    print(f"  Train: {train_X.shape}")
    print(f"  Val:   {val_X.shape}")
    print(f"  Test:  {test_X.shape}")

    # Print class distribution
    print(f"\nClass distribution (train):")
    for i, name in enumerate(config.PUNCH_CLASSES):
        count = (train_y == i).sum()
        if count > 0:
            print(f"  {name}: {count}")

    np.savez(output_dir / "punch_train.npz", X=train_X, y=train_y)
    np.savez(output_dir / "punch_val.npz", X=val_X, y=val_y)
    np.savez(output_dir / "punch_test.npz", X=test_X, y=test_y)

    print(f"\nSaved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Load BoxingVI dataset and create splits.")
    parser.add_argument("--output_dir", type=str, default=str(config.SPLITS_DIR),
                        help="Output directory for .npz splits")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable augmentation")
    args = parser.parse_args()

    build_splits(Path(args.output_dir), augment=not args.no_augment)


if __name__ == "__main__":
    main()
