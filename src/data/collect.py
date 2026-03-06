"""
Webcam recording tool for data collection.

Usage:
    python -m src.data.collect --action jab_left --output_dir data/raw

Controls:
    SPACE  — Start/stop recording a clip
    Q/ESC  — Quit
    S      — Save current clip
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

import config


def create_output_dirs():
    """Ensure output directories exist."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)


def run_collection(action_type: str, camera_index: int = config.CAMERA_INDEX):
    """
    Launch webcam preview with recording controls.

    Args:
        action_type: Label for the action being recorded (e.g. 'jab_left', 'duck').
        camera_index: Webcam device index.
    """
    create_output_dirs()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    print(f"Camera opened: {actual_w}x{actual_h} @ {fps}fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    recording = False
    writer = None
    clip_path = None
    clip_count = 0
    log_path = config.ANNOTATIONS_DIR / f"collection_log_{action_type}.csv"

    # Append to existing log or create new
    log_exists = log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if not log_exists:
        log_writer.writerow(["filename", "action_type", "timestamp", "fps", "width", "height"])

    print(f"\nCollecting: {action_type}")
    print("Controls: SPACE=start/stop recording | S=save clip | Q/ESC=quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Draw status overlay
            display = frame.copy()
            status = "RECORDING" if recording else "PREVIEW"
            color = (0, 0, 255) if recording else (0, 255, 0)
            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(display, f"Action: {action_type}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Clips saved: {clip_count}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Boxing Trainer - Data Collection", display)

            if recording and writer is not None:
                writer.write(frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):  # SPACE — toggle recording
                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{action_type}_{timestamp}.mp4"
                    clip_path = config.RAW_DIR / filename
                    writer = cv2.VideoWriter(str(clip_path), fourcc, fps,
                                             (actual_w, actual_h))
                    recording = True
                    print(f"  Recording started: {filename}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"  Recording stopped: {clip_path.name}")

            elif key == ord("s"):  # S — save and finalize clip
                if recording:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None

                if clip_path and clip_path.exists():
                    log_writer.writerow([
                        clip_path.name, action_type,
                        datetime.now().isoformat(), fps, actual_w, actual_h
                    ])
                    log_file.flush()
                    clip_count += 1
                    print(f"  Saved clip #{clip_count}: {clip_path.name}")
                    clip_path = None

            elif key in (ord("q"), 27):  # Q or ESC — quit
                break

    finally:
        if recording and writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\nDone. {clip_count} clips saved for '{action_type}'.")


def main():
    parser = argparse.ArgumentParser(description="Record boxing action clips.")
    parser.add_argument("--action", type=str, required=True,
                        help="Action type to record (e.g. jab_left, duck)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Webcam device index")
    args = parser.parse_args()

    valid_actions = config.PUNCH_CLASSES + config.DEFENSE_CLASSES
    if args.action not in valid_actions:
        print(f"Warning: '{args.action}' not in known actions: {valid_actions}")
        print("Proceeding anyway...\n")

    run_collection(args.action, args.camera)


if __name__ == "__main__":
    main()
