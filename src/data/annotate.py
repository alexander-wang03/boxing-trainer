"""
Annotation helper for reviewing recorded clips and labeling actions.

Usage:
    python -m src.data.annotate --input data/raw/

Controls:
    SPACE      — Play/pause
    LEFT/RIGHT — Step backward/forward one frame
    S          — Mark start frame of action
    E          — Mark end frame of action
    ENTER      — Save annotation and move to next clip
    Q/ESC      — Quit
"""

import argparse
import csv
from pathlib import Path

import cv2

import config


def annotate_videos(input_dir: Path = config.RAW_DIR,
                    output_csv: Path = config.ANNOTATIONS_DIR / "annotations.csv"):
    """
    Interactive annotation tool: review each video clip and mark action boundaries.
    """
    config.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(input_dir.glob("*.mp4")) +
        list(input_dir.glob("*.avi")) +
        list(input_dir.glob("*.mov"))
    )

    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    # Load existing annotations to skip already-annotated clips
    existing = set()
    if output_csv.exists():
        with open(output_csv, "r") as f:
            reader = csv.DictReader(f)
            existing = {row["filename"] for row in reader}

    csv_exists = output_csv.exists()
    csv_file = open(output_csv, "a", newline="")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(["filename", "action_type", "hand", "start_frame", "end_frame",
                          "total_frames"])

    pending = [v for v in video_files if v.name not in existing]
    print(f"\n{len(pending)} clips to annotate ({len(existing)} already done).\n")

    for idx, video_path in enumerate(pending):
        print(f"\n[{idx + 1}/{len(pending)}] {video_path.name}")

        # Infer action type from filename (e.g. jab_left_20260301_120000.mp4)
        parts = video_path.stem.split("_")
        if len(parts) >= 2:
            # Try to reconstruct action type (e.g. "jab_left")
            action_guess = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
        else:
            action_guess = "unknown"

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Error: Cannot open {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            print(f"  Empty video, skipping.")
            continue

        current_frame = 0
        start_frame = -1
        end_frame = -1
        playing = False

        print(f"  Frames: {len(frames)}, FPS: {fps:.1f}")
        print(f"  Guessed action: {action_guess}")
        print("  Controls: SPACE=play/pause, S=start, E=end, ENTER=save, Q=quit")

        while True:
            display = frames[current_frame].copy()

            # Draw frame info
            cv2.putText(display, f"Frame: {current_frame}/{len(frames) - 1}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(display, f"Action: {action_guess}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            start_text = f"Start: {start_frame}" if start_frame >= 0 else "Start: [press S]"
            end_text = f"End: {end_frame}" if end_frame >= 0 else "End: [press E]"
            cv2.putText(display, start_text, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, end_text, (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Annotate", display)

            if playing:
                current_frame = min(current_frame + 1, len(frames) - 1)
                if current_frame == len(frames) - 1:
                    playing = False
                wait_ms = max(1, int(1000 / fps))
            else:
                wait_ms = 0  # Block until keypress

            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord(" "):
                playing = not playing
            elif key == 81 or key == 2:  # LEFT arrow
                playing = False
                current_frame = max(0, current_frame - 1)
            elif key == 83 or key == 3:  # RIGHT arrow
                playing = False
                current_frame = min(len(frames) - 1, current_frame + 1)
            elif key == ord("s"):
                start_frame = current_frame
                print(f"  Start frame set: {start_frame}")
            elif key == ord("e"):
                end_frame = current_frame
                print(f"  End frame set: {end_frame}")
            elif key == 13:  # ENTER — save
                if start_frame >= 0 and end_frame >= 0 and end_frame > start_frame:
                    # Determine hand from action name
                    hand = ""
                    if "left" in action_guess:
                        hand = "left"
                    elif "right" in action_guess:
                        hand = "right"

                    writer.writerow([
                        video_path.name, action_guess, hand,
                        start_frame, end_frame, len(frames)
                    ])
                    csv_file.flush()
                    print(f"  Saved: {action_guess} frames [{start_frame}–{end_frame}]")
                    break
                else:
                    print("  Error: Set valid start (S) and end (E) frames first.")
            elif key in (ord("q"), 27):
                cv2.destroyAllWindows()
                csv_file.close()
                print("Quit.")
                return

        cv2.destroyAllWindows()

    csv_file.close()
    print(f"\nAnnotation complete. Saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Annotate recorded boxing clips.")
    parser.add_argument("--input", type=str, default=str(config.RAW_DIR),
                        help="Directory with video clips")
    parser.add_argument("--output", type=str,
                        default=str(config.ANNOTATIONS_DIR / "annotations.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    annotate_videos(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
