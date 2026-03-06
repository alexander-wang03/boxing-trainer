"""
Preprocessing pipeline: windowing, augmentation, and train/val/test splitting.

Usage:
    python -m src.data.preprocess
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import config


def create_windows(keypoints: np.ndarray, window_size: int = config.SEQUENCE_LENGTH,
                   stride: int = 15) -> list[np.ndarray]:
    """
    Create sliding windows from a keypoint sequence.

    Args:
        keypoints: Shape (num_frames, 33, 3).
        window_size: Number of frames per window.
        stride: Step size between windows.

    Returns:
        List of arrays, each shape (window_size, 33, 3).
    """
    windows = []
    num_frames = len(keypoints)

    if num_frames < window_size:
        # Pad with last frame if too short
        pad_count = window_size - num_frames
        padding = np.tile(keypoints[-1:], (pad_count, 1, 1))
        padded = np.concatenate([keypoints, padding], axis=0)
        windows.append(padded)
    else:
        for start in range(0, num_frames - window_size + 1, stride):
            windows.append(keypoints[start:start + window_size])

    return windows


def augment_horizontal_flip(window: np.ndarray) -> np.ndarray:
    """
    Mirror keypoints horizontally (flip x-coordinates).
    Also swaps left/right keypoint pairs per MediaPipe Pose topology.

    Args:
        window: Shape (seq_len, 33, 3).

    Returns:
        Flipped array of same shape.
    """
    flipped = window.copy()
    # Flip x-coordinate (already normalized, so negate x)
    flipped[:, :, 0] *= -1

    # Swap left/right landmark pairs (MediaPipe Pose)
    swap_pairs = [
        (1, 4), (2, 5), (3, 6),      # eyes, ears
        (7, 8),                        # ear tips (inner)
        (9, 10),                       # mouth
        (11, 12),                      # shoulders
        (13, 14),                      # elbows
        (15, 16),                      # wrists
        (17, 18), (19, 20), (21, 22),  # hands
        (23, 24),                      # hips
        (25, 26),                      # knees
        (27, 28),                      # ankles
        (29, 30), (31, 32),            # feet
    ]
    for left_idx, right_idx in swap_pairs:
        flipped[:, [left_idx, right_idx]] = flipped[:, [right_idx, left_idx]]

    return flipped


def augment_speed_variation(window: np.ndarray,
                            speed_range: tuple = config.SPEED_VARIATION_RANGE) -> np.ndarray:
    """
    Resample window at a random speed factor via linear interpolation.

    Args:
        window: Shape (seq_len, 33, 3).
        speed_range: (min_speed, max_speed) multiplier.

    Returns:
        Resampled array of shape (seq_len, 33, 3).
    """
    seq_len = len(window)
    speed_factor = np.random.uniform(*speed_range)
    new_len = int(seq_len * speed_factor)

    if new_len < 2:
        return window

    # Interpolate each keypoint dimension
    original_indices = np.linspace(0, seq_len - 1, new_len)
    target_indices = np.linspace(0, seq_len - 1, seq_len)

    resampled = np.zeros_like(window)
    for kp in range(window.shape[1]):
        for dim in range(window.shape[2]):
            resampled[:, kp, dim] = np.interp(target_indices, original_indices,
                                               np.interp(original_indices,
                                                         np.arange(seq_len),
                                                         window[:, kp, dim]))

    return resampled


def augment_frame_drop(window: np.ndarray,
                       drop_prob: float = config.FRAME_DROP_PROB) -> np.ndarray:
    """
    Randomly drop frames and fill with linear interpolation.

    Args:
        window: Shape (seq_len, 33, 3).
        drop_prob: Probability of dropping each frame.

    Returns:
        Array of same shape with dropped frames interpolated.
    """
    seq_len = len(window)
    mask = np.random.random(seq_len) > drop_prob
    # Always keep first and last frame
    mask[0] = True
    mask[-1] = True

    if mask.sum() == seq_len:
        return window

    result = window.copy()
    kept_indices = np.where(mask)[0]

    for kp in range(window.shape[1]):
        for dim in range(window.shape[2]):
            result[:, kp, dim] = np.interp(
                np.arange(seq_len), kept_indices, window[kept_indices, kp, dim]
            )

    return result


def extract_head_features(window: np.ndarray) -> np.ndarray:
    """
    Extract head-specific features for the defense classifier.

    Features per frame: head keypoint positions + velocity.

    Args:
        window: Shape (seq_len, 33, 3).

    Returns:
        Array of shape (seq_len, num_head_features).
    """
    head_kps = window[:, config.HEAD_KEYPOINT_INDICES, :]  # (seq_len, 11, 3)
    seq_len = head_kps.shape[0]

    # Flatten head positions
    positions = head_kps.reshape(seq_len, -1)  # (seq_len, 33)

    # Compute velocity (frame-to-frame difference)
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    # Concatenate position + velocity
    features = np.concatenate([positions, velocity], axis=1)  # (seq_len, 66)

    return features.astype(np.float32)


def flatten_window_for_punch(window: np.ndarray) -> np.ndarray:
    """
    Flatten window to (seq_len, 99) for punch classifier input.

    Args:
        window: Shape (seq_len, 33, 3).

    Returns:
        Array of shape (seq_len, 99).
    """
    seq_len = window.shape[0]
    return window.reshape(seq_len, -1).astype(np.float32)


def get_flipped_label(label: str) -> str:
    """Swap left/right in a label after horizontal flip."""
    if "left" in label:
        return label.replace("left", "right")
    elif "right" in label:
        return label.replace("right", "left")
    return label


def build_dataset(annotations_csv: Path = config.ANNOTATIONS_DIR / "annotations.csv",
                  processed_dir: Path = config.PROCESSED_DIR,
                  output_dir: Path = config.SPLITS_DIR,
                  augment: bool = True):
    """
    Build train/val/test splits from annotations and extracted keypoints.

    Saves:
        - punch_train.npz, punch_val.npz, punch_test.npz
        - defense_train.npz, defense_val.npz, defense_test.npz
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_csv.exists():
        print(f"Error: Annotations file not found: {annotations_csv}")
        return

    # Load annotations
    with open(annotations_csv, "r") as f:
        reader = csv.DictReader(f)
        annotations = list(reader)

    print(f"Loaded {len(annotations)} annotations.")

    punch_sequences = []
    punch_labels = []
    defense_sequences = []
    defense_labels = []

    for ann in annotations:
        filename = ann["filename"]
        action = ann["action_type"]
        start = int(ann["start_frame"])
        end = int(ann["end_frame"])

        npy_path = processed_dir / f"{Path(filename).stem}.npy"
        if not npy_path.exists():
            print(f"  Warning: {npy_path} not found, skipping.")
            continue

        keypoints = np.load(npy_path)
        clip = keypoints[start:end + 1]

        # Determine if this is a punch or defense action
        is_punch = action in config.PUNCH_CLASSES
        is_defense = action in config.DEFENSE_CLASSES

        if not is_punch and not is_defense:
            print(f"  Warning: Unknown action '{action}', skipping.")
            continue

        # Create windows from clip
        windows = create_windows(clip)

        for window in windows:
            if is_punch:
                punch_sequences.append(flatten_window_for_punch(window))
                punch_labels.append(config.PUNCH_CLASSES.index(action))

                # Augmentation
                if augment:
                    # Horizontal flip
                    flipped = augment_horizontal_flip(window)
                    flipped_label = get_flipped_label(action)
                    if flipped_label in config.PUNCH_CLASSES:
                        punch_sequences.append(flatten_window_for_punch(flipped))
                        punch_labels.append(config.PUNCH_CLASSES.index(flipped_label))

                    # Speed variation
                    speed_aug = augment_speed_variation(window)
                    punch_sequences.append(flatten_window_for_punch(speed_aug))
                    punch_labels.append(config.PUNCH_CLASSES.index(action))

                    # Frame drop
                    dropped = augment_frame_drop(window)
                    punch_sequences.append(flatten_window_for_punch(dropped))
                    punch_labels.append(config.PUNCH_CLASSES.index(action))

            elif is_defense:
                defense_sequences.append(extract_head_features(window))
                defense_labels.append(config.DEFENSE_CLASSES.index(action))

                if augment:
                    # Speed variation
                    speed_aug = augment_speed_variation(window)
                    defense_sequences.append(extract_head_features(speed_aug))
                    defense_labels.append(config.DEFENSE_CLASSES.index(action))

                    # Frame drop
                    dropped = augment_frame_drop(window)
                    defense_sequences.append(extract_head_features(dropped))
                    defense_labels.append(config.DEFENSE_CLASSES.index(action))

    # Convert to arrays and split
    for name, sequences, labels in [
        ("punch", punch_sequences, punch_labels),
        ("defense", defense_sequences, defense_labels),
    ]:
        if not sequences:
            print(f"  No {name} data found, skipping split.")
            continue

        X = np.array(sequences)
        y = np.array(labels)
        print(f"\n{name.upper()} dataset: {X.shape[0]} samples, shape {X.shape[1:]}")

        # Stratified split: train / (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(config.VAL_RATIO + config.TEST_RATIO),
            stratify=y, random_state=42
        )
        # Split temp into val / test
        relative_test = config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=relative_test,
            stratify=y_temp, random_state=42
        )

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        np.savez(output_dir / f"{name}_train.npz", X=X_train, y=y_train)
        np.savez(output_dir / f"{name}_val.npz", X=X_val, y=y_val)
        np.savez(output_dir / f"{name}_test.npz", X=X_test, y=y_test)

    print(f"\nSplits saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split dataset.")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    args = parser.parse_args()

    build_dataset(augment=not args.no_augment)


if __name__ == "__main__":
    main()
