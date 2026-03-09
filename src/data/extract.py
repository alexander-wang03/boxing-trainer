"""
MediaPipe keypoint extraction from recorded video clips.

Usage:
    python -m src.data.extract
    python -m src.data.extract --input data/raw/jab_left_20260301.mp4

Extracts 33 keypoints x 3 coordinates per frame, normalizes to shoulder width,
and saves as .npy arrays in data/processed/.
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_tasks_vision
from tqdm import tqdm

import config


# MediaPipe Pose landmark indices for shoulders
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Pose landmarker model
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_MODEL_PATH = config.PROJECT_ROOT / "pose_landmarker_full.task"


def ensure_model() -> Path:
    """Download pose landmarker model file if not already present."""
    if not _MODEL_PATH.exists():
        print(f"Downloading pose landmarker model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")
    return _MODEL_PATH


def extract_keypoints_from_result(mp_result) -> np.ndarray | None:
    """
    Extract 33 keypoints (x, y, z) from a PoseLandmarker result.

    Returns:
        Array of shape (33, 3) or None if no pose detected.
    """
    if not mp_result.pose_landmarks:
        return None
    landmarks = mp_result.pose_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize keypoints for translation and scale invariance.

    - Translates so midpoint of shoulders is at origin.
    - Scales by shoulder width (distance between shoulders).

    Args:
        keypoints: Array of shape (num_frames, 33, 3).

    Returns:
        Normalized array of same shape.
    """
    normalized = keypoints.copy()

    for i in range(len(normalized)):
        left_shoulder = normalized[i, LEFT_SHOULDER_IDX]
        right_shoulder = normalized[i, RIGHT_SHOULDER_IDX]

        # Translation: center on shoulder midpoint
        midpoint = (left_shoulder + right_shoulder) / 2.0
        normalized[i] -= midpoint

        # Scale: normalize by shoulder width
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if shoulder_width > 1e-6:
            normalized[i] /= shoulder_width

    return normalized


def extract_from_video(video_path: Path, normalize: bool = True) -> np.ndarray | None:
    """
    Extract and optionally normalize keypoints from an entire video.

    Args:
        video_path: Path to video file.
        normalize: Whether to apply shoulder-width normalization.

    Returns:
        Array of shape (num_frames, 33, 3) or None if extraction fails.
    """
    model_path = ensure_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    base_options = mp_tasks_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_tasks_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_tasks_vision.RunningMode.VIDEO,
    )

    all_keypoints = []
    frames_with_no_pose = 0

    with mp_tasks_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            kp = extract_keypoints_from_result(result)
            if kp is not None:
                all_keypoints.append(kp)
            else:
                frames_with_no_pose += 1
                # Fill with previous frame's keypoints or zeros
                if all_keypoints:
                    all_keypoints.append(all_keypoints[-1].copy())
                else:
                    all_keypoints.append(
                        np.zeros((config.NUM_KEYPOINTS, config.KEYPOINT_DIMS), dtype=np.float32)
                    )

            frame_idx += 1

    cap.release()

    if not all_keypoints:
        print(f"Warning: No keypoints extracted from {video_path}")
        return None

    keypoints_array = np.array(all_keypoints, dtype=np.float32)

    if frames_with_no_pose > 0:
        pct = frames_with_no_pose / len(all_keypoints) * 100
        print(f"  {video_path.name}: {frames_with_no_pose}/{len(all_keypoints)} "
              f"frames ({pct:.1f}%) had no pose detected (filled with previous)")

    if normalize:
        keypoints_array = normalize_keypoints(keypoints_array)

    return keypoints_array


def process_all_videos(input_dir: Path = config.RAW_DIR,
                       output_dir: Path = config.PROCESSED_DIR):
    """
    Extract keypoints from all videos in input_dir and save as .npy files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(input_dir.glob("*.mp4")) +
        list(input_dir.glob("*.avi")) +
        list(input_dir.glob("*.mov"))
    )

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Processing {len(video_files)} videos from {input_dir}...")

    for video_path in tqdm(video_files, desc="Extracting keypoints"):
        output_path = output_dir / f"{video_path.stem}.npy"

        if output_path.exists():
            print(f"  Skipping {video_path.name} (already processed)")
            continue

        keypoints = extract_from_video(video_path)

        if keypoints is not None:
            np.save(output_path, keypoints)
            print(f"  {video_path.name} -> {output_path.name} "
                  f"(shape: {keypoints.shape})")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Extract MediaPipe keypoints from videos.")
    parser.add_argument("--input", type=str, default=None,
                        help="Single video file to process (default: all in data/raw/)")
    parser.add_argument("--output_dir", type=str, default=str(config.PROCESSED_DIR),
                        help="Output directory for .npy files")
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip shoulder-width normalization")
    args = parser.parse_args()

    if args.input:
        video_path = Path(args.input)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        keypoints = extract_from_video(video_path, normalize=not args.no_normalize)
        if keypoints is not None:
            out_path = output_dir / f"{video_path.stem}.npy"
            np.save(out_path, keypoints)
            print(f"Saved: {out_path} (shape: {keypoints.shape})")
    else:
        process_all_videos(output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
