"""
Webcam recording tool for data collection.

Manual mode:
    python -m src.data.collect --action jab_left

Automated mode (cycles through all classes, no keyboard input needed):
    python -m src.data.collect --auto
    python -m src.data.collect --auto --clips 100 --prep 2.5 --record 1.0 --rest 0.5

Manual controls:
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
import numpy as np

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


def _draw_auto_hud(frame: np.ndarray, action: str, next_action: str | None,
                   phase: str, phase_progress: float,
                   clip_idx: int, clips_per_class: int,
                   class_idx: int, total_classes: int) -> np.ndarray:
    """
    Overlay HUD elements for automated collection mode.

    phase:          'prep' | 'record' | 'rest' | 'between'
    phase_progress: 0.0 → 1.0 fraction of current phase elapsed
    """
    h, w = frame.shape[:2]
    out = frame.copy()

    # Dim the frame slightly so text is always readable
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    # ── Phase colour and label ──────────────────────────────────────────
    phase_cfg = {
        "prep":    ((0, 200, 255), "GET READY"),   # amber
        "record":  ((0, 0, 255),   "GO!"),          # red
        "rest":    ((0, 220, 0),   "REST"),          # green
        "between": ((180, 0, 255), "NEXT CLASS"),   # purple
    }
    bar_color, phase_label = phase_cfg.get(phase, ((200, 200, 200), phase.upper()))

    # ── Big action name (centre) ────────────────────────────────────────
    action_display = action.replace("_", " ").upper()
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(action_display, font, 2.2, 3)
    tx = (w - text_size[0]) // 2
    ty = h // 2 + text_size[1] // 2
    cv2.putText(out, action_display, (tx, ty), font, 2.2, (255, 255, 255), 3)

    # ── Phase label (above action name) ────────────────────────────────
    pl_size, _ = cv2.getTextSize(phase_label, font, 1.2, 2)
    cv2.putText(out, phase_label,
                ((w - pl_size[0]) // 2, ty - text_size[1] - 20),
                font, 1.2, bar_color, 2)

    # ── Progress bar (bottom of screen) ────────────────────────────────
    bar_h = 18
    bar_y = h - 40
    bar_x0, bar_x1 = 40, w - 40
    cv2.rectangle(out, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (60, 60, 60), -1)
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * phase_progress)
    cv2.rectangle(out, (bar_x0, bar_y), (fill_x, bar_y + bar_h), bar_color, -1)

    # ── Clip counter (top-left) ─────────────────────────────────────────
    cv2.putText(out, f"Clip  {clip_idx}/{clips_per_class}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"Class {class_idx}/{total_classes}",
                (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # ── Next class (top-right) ──────────────────────────────────────────
    if next_action:
        next_label = "Next: " + next_action.replace("_", " ").upper()
        nl_size, _ = cv2.getTextSize(next_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.putText(out, next_label, (w - nl_size[0] - 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2)

    return out


def run_auto_collection(
    actions: list[str] | None = None,
    clips_per_class: int = 100,
    prep_seconds: float = 2.5,
    record_seconds: float = 1.0,
    rest_seconds: float = 0.5,
    between_class_seconds: float = 3.0,
    camera_index: int = config.CAMERA_INDEX,
):
    """
    Fully automated data collection: cycles through every action class,
    records clips on a fixed timer, requires no keyboard input.

    Timeline per clip:
        [prep_seconds] GET READY → [record_seconds] GO! → [rest_seconds] REST
    Between classes:
        [between_class_seconds] NEXT CLASS pause

    Press ESC at any time to stop early.
    """
    if actions is None:
        # Skip 'neutral' — it'll be sampled from non-action segments at extract time
        actions = [a for a in config.PUNCH_CLASSES if a != "neutral"] + \
                  [a for a in config.DEFENSE_CLASSES if a != "neutral"]

    create_output_dirs()

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or config.FRAME_RATE
    print(f"Camera: {actual_w}x{actual_h} @ {cam_fps:.0f}fps")
    print(f"Classes: {len(actions)}  |  Clips/class: {clips_per_class}")
    print(f"Timing: prep={prep_seconds}s  record={record_seconds}s  rest={rest_seconds}s")
    print("Press ESC to stop early.\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    total_saved = 0

    # Pre-open one log file for the whole session
    log_path = config.ANNOTATIONS_DIR / f"auto_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["filename", "action_type", "timestamp", "fps", "width", "height"])

    try:
        for class_idx, action in enumerate(actions, start=1):
            next_action = actions[class_idx] if class_idx < len(actions) else None
            clip_idx = 0

            # ── Between-class pause ──────────────────────────────────────
            phase_start = time.time()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                elapsed = time.time() - phase_start
                if elapsed >= between_class_seconds:
                    break
                hud = _draw_auto_hud(frame, action, next_action,
                                     "between", elapsed / between_class_seconds,
                                     clip_idx, clips_per_class,
                                     class_idx, len(actions))
                cv2.imshow("Boxing Trainer - Auto Collection", hud)
                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt

            while clip_idx < clips_per_class:
                # ── PREP phase ───────────────────────────────────────────
                phase_start = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    elapsed = time.time() - phase_start
                    if elapsed >= prep_seconds:
                        break
                    hud = _draw_auto_hud(frame, action, next_action,
                                         "prep", elapsed / prep_seconds,
                                         clip_idx, clips_per_class,
                                         class_idx, len(actions))
                    cv2.imshow("Boxing Trainer - Auto Collection", hud)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt

                # ── RECORD phase ─────────────────────────────────────────
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{action}_{timestamp}.mp4"
                clip_path = config.RAW_DIR / filename
                writer = cv2.VideoWriter(str(clip_path), fourcc, cam_fps,
                                         (actual_w, actual_h))
                phase_start = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    elapsed = time.time() - phase_start
                    if elapsed >= record_seconds:
                        break
                    writer.write(frame)
                    hud = _draw_auto_hud(frame, action, next_action,
                                         "record", elapsed / record_seconds,
                                         clip_idx + 1, clips_per_class,
                                         class_idx, len(actions))
                    cv2.imshow("Boxing Trainer - Auto Collection", hud)
                    if cv2.waitKey(1) & 0xFF == 27:
                        writer.release()
                        raise KeyboardInterrupt
                writer.release()

                # Log it
                log_writer.writerow([filename, action, datetime.now().isoformat(),
                                      cam_fps, actual_w, actual_h])
                log_file.flush()
                clip_idx += 1
                total_saved += 1
                print(f"  [{action}] clip {clip_idx}/{clips_per_class}  (total {total_saved})")

                # ── REST phase ───────────────────────────────────────────
                phase_start = time.time()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    elapsed = time.time() - phase_start
                    if elapsed >= rest_seconds:
                        break
                    hud = _draw_auto_hud(frame, action, next_action,
                                         "rest", elapsed / rest_seconds,
                                         clip_idx, clips_per_class,
                                         class_idx, len(actions))
                    cv2.imshow("Boxing Trainer - Auto Collection", hud)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nStopped early by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log_file.close()
        print(f"\nDone. {total_saved} clips saved. Log: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Record boxing action clips.")
    submode = parser.add_mutually_exclusive_group(required=True)
    submode.add_argument("--action", type=str,
                         help="Manual mode: action type to record (e.g. jab_left, duck)")
    submode.add_argument("--auto", action="store_true",
                         help="Automated mode: cycles through all classes automatically")

    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)

    # Auto-mode options
    parser.add_argument("--clips", type=int, default=100,
                        help="Clips to record per class (auto mode, default 100)")
    parser.add_argument("--prep", type=float, default=2.5,
                        help="Seconds to show GET READY before recording (default 2.5)")
    parser.add_argument("--record", type=float, default=1.0,
                        help="Seconds to record each clip (default 1.0)")
    parser.add_argument("--rest", type=float, default=0.5,
                        help="Rest seconds between clips (default 0.5)")
    parser.add_argument("--between", type=float, default=3.0,
                        help="Pause seconds between classes (default 3.0)")
    parser.add_argument("--actions", type=str, nargs="+", default=None,
                        help="Subset of actions to record (auto mode only, default: all non-neutral)")

    args = parser.parse_args()

    if args.auto:
        run_auto_collection(
            actions=args.actions,
            clips_per_class=args.clips,
            prep_seconds=args.prep,
            record_seconds=args.record,
            rest_seconds=args.rest,
            between_class_seconds=args.between,
            camera_index=args.camera,
        )
    else:
        valid_actions = config.PUNCH_CLASSES + config.DEFENSE_CLASSES
        if args.action not in valid_actions:
            print(f"Warning: '{args.action}' not in known actions: {valid_actions}")
            print("Proceeding anyway...\n")
        run_collection(args.action, args.camera)


if __name__ == "__main__":
    main()
