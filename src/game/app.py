"""
Main application entry point: Pygame game loop with webcam + inference.

Usage:
    python -m src.game.app
    python -m src.game.app --camera 0 --difficulty medium
"""

import argparse
import sys

import cv2
import pygame

import config
from src.game.game_logic import GameLogic, GameState
from src.game.inference import RealtimeInference
from src.game.renderer import Renderer


class BoxingTrainerApp:
    """
    Main application class tying together webcam capture,
    real-time inference, game logic, and Pygame rendering.
    """

    def __init__(self, camera_index: int = config.CAMERA_INDEX,
                 difficulty: str = "medium",
                 punch_checkpoint: str | None = None,
                 defense_checkpoint: str | None = None):
        self.camera_index = camera_index
        self.difficulty = difficulty

        # Components
        self.cap: cv2.VideoCapture | None = None
        self.inference = RealtimeInference(
            punch_checkpoint=punch_checkpoint,
            defense_checkpoint=defense_checkpoint,
        )
        self.game = GameLogic()
        self.renderer = Renderer()
        self.clock = pygame.time.Clock()

        self.running = False
        self.target_fps = config.FRAME_RATE

    def init(self):
        """Initialize Pygame and webcam."""
        pygame.init()
        self.renderer.init()

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        if not self.cap.isOpened():
            print("Error: Cannot open webcam.")
            sys.exit(1)

        self.game.set_difficulty(self.difficulty)
        self.running = True

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera: {actual_w}x{actual_h}")
        print(f"Difficulty: {self.difficulty}")
        print("Ready. Press SPACE to start.\n")

    def run(self):
        """Main game loop."""
        self.init()

        while self.running:
            self._handle_events()

            if self.game.state == GameState.MENU:
                self.renderer.render_menu()

            elif self.game.state == GameState.COUNTDOWN:
                remaining = self.game.get_countdown_remaining()
                self.renderer.render_countdown(remaining)
                self.game.update(None, None)

            elif self.game.state == GameState.TRAINING:
                self._training_loop()

            elif self.game.state == GameState.ROUND_END:
                self.renderer.render_round_end(self.game)

            pygame.display.flip()
            self.clock.tick(self.target_fps)

        self.cleanup()

    def _training_loop(self):
        """Process one frame during active training."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Run inference
        result = self.inference.process_frame(frame)

        # Draw skeleton on frame
        annotated = self.inference.draw_skeleton(frame, result["pose_landmarks"])

        # Update game logic
        prev_cue = self.game.active_cue
        self.game.update(result["punch_class"], result["defense_class"])

        # Trigger visual feedback on cue response
        if prev_cue and prev_cue.responded and prev_cue.correct:
            self.renderer.trigger_flash(correct=True)
        elif prev_cue and not self.game.active_cue and not prev_cue.responded:
            self.renderer.trigger_flash(correct=False)

        # Render
        self.renderer.render_training(
            annotated, self.game,
            result["punch_class"], result["punch_confidence"],
            result["defense_class"], result["defense_confidence"],
            result["inference_ms"],
        )

    def _handle_events(self):
        """Handle Pygame events (keyboard, quit)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    if self.game.state == GameState.MENU:
                        self.running = False
                    else:
                        self.game.state = GameState.ROUND_END

                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if self.game.state == GameState.MENU:
                        self.game.start_round()
                    elif self.game.state == GameState.ROUND_END:
                        self.game.next_round()

                elif event.key == pygame.K_1:
                    self.game.set_difficulty("easy")
                    print("Difficulty: Easy")
                elif event.key == pygame.K_2:
                    self.game.set_difficulty("medium")
                    print("Difficulty: Medium")
                elif event.key == pygame.K_3:
                    self.game.set_difficulty("hard")
                    print("Difficulty: Hard")

    def cleanup(self):
        """Release all resources."""
        if self.cap:
            self.cap.release()
        self.inference.cleanup()
        pygame.quit()
        print("\nSession ended.")


def main():
    parser = argparse.ArgumentParser(description="Shadow Boxing Trainer")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                        help="Webcam device index")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard"],
                        help="Game difficulty")
    parser.add_argument("--punch-checkpoint", type=str, default=None,
                        help="Path to punch model checkpoint")
    parser.add_argument("--defense-checkpoint", type=str, default=None,
                        help="Path to defense model checkpoint")
    args = parser.parse_args()

    app = BoxingTrainerApp(
        camera_index=args.camera,
        difficulty=args.difficulty,
        punch_checkpoint=args.punch_checkpoint,
        defense_checkpoint=args.defense_checkpoint,
    )
    app.run()


if __name__ == "__main__":
    main()
