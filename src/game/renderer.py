"""
Pygame renderer: webcam feed, skeleton overlay, HUD, cues, and menus.
"""

import time

import cv2
import numpy as np
import pygame

import config
from src.game.game_logic import GameLogic, GameState


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 220)
YELLOW = (255, 220, 50)
ORANGE = (255, 150, 30)
GRAY = (120, 120, 120)
DARK_BG = (20, 20, 30)
SEMI_TRANSPARENT = (0, 0, 0, 150)


class Renderer:
    """
    Handles all Pygame drawing: webcam feed, overlays, HUD, cues, and menus.
    """

    def __init__(self, width: int = config.WINDOW_WIDTH,
                 height: int = config.WINDOW_HEIGHT):
        self.width = width
        self.height = height
        self.screen: pygame.Surface | None = None

        # Fonts (initialized after pygame.init)
        self.font_large: pygame.font.Font | None = None
        self.font_medium: pygame.font.Font | None = None
        self.font_small: pygame.font.Font | None = None
        self.font_title: pygame.font.Font | None = None

        # Visual feedback
        self.flash_color: tuple | None = None
        self.flash_start = 0.0
        self.flash_duration = 0.3

    def init(self):
        """Initialize Pygame display and fonts."""
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Shadow Boxing Trainer")

        self.font_title = pygame.font.SysFont("Arial", 64, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 32)
        self.font_small = pygame.font.SysFont("Arial", 22)

    def frame_to_surface(self, frame: np.ndarray) -> pygame.Surface:
        """Convert an OpenCV BGR frame to a Pygame surface, scaled to window size."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to window dimensions
        rgb = cv2.resize(rgb, (self.width, self.height))
        # OpenCV uses (H, W, C), Pygame expects (W, H) surface
        surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        return surface

    def render_menu(self):
        """Draw the main menu screen."""
        self.screen.fill(DARK_BG)

        # Title
        title = self.font_title.render("SHADOW BOXING", True, WHITE)
        subtitle = self.font_large.render("TRAINER", True, YELLOW)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 3 - 30))
        sub_rect = subtitle.get_rect(center=(self.width // 2, self.height // 3 + 40))
        self.screen.blit(title, title_rect)
        self.screen.blit(subtitle, sub_rect)

        # Instructions
        instructions = [
            "Press SPACE or ENTER to start training",
            "Press 1/2/3 to set difficulty (Easy/Medium/Hard)",
            "Press Q or ESC to quit",
        ]
        y = self.height // 2 + 40
        for text in instructions:
            rendered = self.font_small.render(text, True, GRAY)
            rect = rendered.get_rect(center=(self.width // 2, y))
            self.screen.blit(rendered, rect)
            y += 35

    def render_countdown(self, seconds_remaining: float):
        """Draw countdown overlay."""
        self.screen.fill(DARK_BG)
        count = max(1, int(seconds_remaining) + 1)
        text = self.font_title.render(str(count), True, YELLOW)
        rect = text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(text, rect)

        label = self.font_medium.render("GET READY!", True, WHITE)
        label_rect = label.get_rect(center=(self.width // 2, self.height // 2 - 80))
        self.screen.blit(label, label_rect)

    def render_training(self, frame: np.ndarray, game: GameLogic,
                        punch_class: str | None, punch_conf: float,
                        defense_class: str | None, defense_conf: float,
                        inference_ms: float):
        """Draw the main training view with webcam feed and HUD."""
        # Webcam background
        surface = self.frame_to_surface(frame)
        self.screen.blit(surface, (0, 0))

        # Flash effect
        self._draw_flash()

        # HUD panel (top bar)
        self._draw_hud(game, inference_ms)

        # Active cue
        if game.active_cue and not game.active_cue.responded:
            self._draw_cue(game)

        # Current detection display
        self._draw_detection(punch_class, punch_conf, defense_class, defense_conf)

        # Combo display
        if game.stats.combo_current > 1:
            self._draw_combo(game.stats.combo_current)

    def render_round_end(self, game: GameLogic):
        """Draw round-end summary screen."""
        self.screen.fill(DARK_BG)

        title = self.font_large.render(f"ROUND {game.stats.round_number} COMPLETE", True, YELLOW)
        title_rect = title.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title, title_rect)

        stats = game.stats
        lines = [
            f"Score: {stats.score}",
            f"Punches Thrown: {stats.total_punches}",
            f"Defense Accuracy: {stats.defense_accuracy:.0%} "
            f"({stats.defenses_correct}/{stats.defenses_total})",
            f"Best Combo: {stats.combo_best}",
        ]

        y = 180
        for line in lines:
            rendered = self.font_medium.render(line, True, WHITE)
            rect = rendered.get_rect(center=(self.width // 2, y))
            self.screen.blit(rendered, rect)
            y += 50

        # Punch breakdown
        y += 20
        header = self.font_small.render("Punch Breakdown:", True, YELLOW)
        self.screen.blit(header, header.get_rect(center=(self.width // 2, y)))
        y += 35

        for punch_type, count in stats.punches_detected.items():
            if punch_type == "neutral" or count == 0:
                continue
            line = self.font_small.render(f"  {punch_type}: {count}", True, GRAY)
            rect = line.get_rect(center=(self.width // 2, y))
            self.screen.blit(line, rect)
            y += 28

        # Continue prompt
        prompt = self.font_small.render("Press SPACE for next round | Q to quit", True, GRAY)
        prompt_rect = prompt.get_rect(center=(self.width // 2, self.height - 60))
        self.screen.blit(prompt, prompt_rect)

    def _draw_hud(self, game: GameLogic, inference_ms: float):
        """Draw the top HUD bar."""
        # Semi-transparent bar
        hud_surface = pygame.Surface((self.width, 60), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 160))
        self.screen.blit(hud_surface, (0, 0))

        # Score
        score_text = self.font_medium.render(f"Score: {game.stats.score}", True, YELLOW)
        self.screen.blit(score_text, (20, 12))

        # Round timer
        remaining = game.get_round_time_remaining()
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        timer_color = RED if remaining < 30 else WHITE
        timer_text = self.font_medium.render(f"{minutes}:{seconds:02d}", True, timer_color)
        timer_rect = timer_text.get_rect(center=(self.width // 2, 30))
        self.screen.blit(timer_text, timer_rect)

        # FPS / latency
        fps_text = self.font_small.render(f"{inference_ms:.0f}ms", True, GREEN)
        self.screen.blit(fps_text, (self.width - 100, 18))

    def _draw_cue(self, game: GameLogic):
        """Draw the active defensive cue prompt."""
        cue = game.active_cue
        if cue is None:
            return

        remaining = game.get_cue_time_remaining()
        urgency = remaining / cue.reaction_window  # 1.0 → 0.0

        # Color shifts from green to red as time runs out
        r = int(220 * (1 - urgency))
        g = int(200 * urgency)
        color = (min(255, r + 50), g, 50)

        # Cue text (large, centered)
        cue_text = self.font_large.render(cue.display_text, True, color)
        cue_rect = cue_text.get_rect(center=(self.width // 2, self.height // 2 - 50))

        # Background box
        bg_rect = cue_rect.inflate(40, 20)
        bg_surface = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, bg_rect.topleft)
        self.screen.blit(cue_text, cue_rect)

        # Timer bar
        bar_width = int(300 * urgency)
        bar_rect = pygame.Rect(
            self.width // 2 - 150,
            self.height // 2,
            bar_width, 8
        )
        pygame.draw.rect(self.screen, color, bar_rect)

    def _draw_detection(self, punch_class: str | None, punch_conf: float,
                        defense_class: str | None, defense_conf: float):
        """Draw current detection results in bottom-left."""
        y = self.height - 80

        if punch_class and punch_class != "neutral":
            text = self.font_small.render(
                f"Punch: {punch_class} ({punch_conf:.0%})", True, ORANGE
            )
            self.screen.blit(text, (20, y))
            y -= 30

        if defense_class and defense_class != "neutral":
            text = self.font_small.render(
                f"Defense: {defense_class} ({defense_conf:.0%})", True, BLUE
            )
            self.screen.blit(text, (20, y))

    def _draw_combo(self, combo: int):
        """Draw combo counter."""
        text = self.font_medium.render(f"COMBO x{combo}", True, YELLOW)
        rect = text.get_rect(topright=(self.width - 20, 70))
        self.screen.blit(text, rect)

    def trigger_flash(self, correct: bool):
        """Trigger a screen flash (green=correct, red=miss)."""
        self.flash_color = GREEN if correct else RED
        self.flash_start = time.time()

    def _draw_flash(self):
        """Draw flash effect if active."""
        if self.flash_color is None:
            return

        elapsed = time.time() - self.flash_start
        if elapsed > self.flash_duration:
            self.flash_color = None
            return

        alpha = int(100 * (1 - elapsed / self.flash_duration))
        flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        flash_surface.fill((*self.flash_color, alpha))
        self.screen.blit(flash_surface, (0, 0))
