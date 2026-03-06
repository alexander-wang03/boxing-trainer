"""
Sparring partner AI, cue generation, and scoring system.

The game logic generates defensive cues (prompts) that the user must respond to,
tracks punches thrown, and manages rounds/scoring.
"""

import random
import time
from dataclasses import dataclass, field
from enum import Enum

import config


class GameState(Enum):
    MENU = "menu"
    COUNTDOWN = "countdown"
    TRAINING = "training"
    ROUND_END = "round_end"
    RESULTS = "results"


@dataclass
class DefenseCue:
    """A defensive prompt the user must respond to."""
    cue_type: str            # e.g. "slip", "duck", "weave", "block"
    display_text: str        # e.g. "SLIP LEFT!", "DUCK!"
    created_at: float        # time.time() when cue was created
    reaction_window: float   # seconds to respond
    responded: bool = False
    correct: bool = False

    @property
    def expired(self) -> bool:
        return time.time() - self.created_at > self.reaction_window


@dataclass
class SessionStats:
    """Statistics for a training session."""
    punches_detected: dict = field(default_factory=lambda: {c: 0 for c in config.PUNCH_CLASSES})
    defenses_correct: int = 0
    defenses_missed: int = 0
    defenses_total: int = 0
    combo_current: int = 0
    combo_best: int = 0
    score: int = 0
    round_number: int = 1

    @property
    def defense_accuracy(self) -> float:
        if self.defenses_total == 0:
            return 0.0
        return self.defenses_correct / self.defenses_total

    @property
    def total_punches(self) -> int:
        return sum(self.punches_detected.values()) - self.punches_detected.get("neutral", 0)


# Cue definitions with display text
CUE_TEMPLATES = [
    {"type": "slip", "text": "SLIP LEFT!"},
    {"type": "slip", "text": "SLIP RIGHT!"},
    {"type": "duck", "text": "DUCK!"},
    {"type": "weave", "text": "WEAVE!"},
    {"type": "block", "text": "BLOCK HIGH!"},
    {"type": "block", "text": "BLOCK!"},
]


class GameLogic:
    """
    Core game logic: manages rounds, cues, scoring, and difficulty.
    """

    def __init__(self, round_duration: float = config.ROUND_DURATION,
                 rest_duration: float = config.REST_DURATION):
        self.round_duration = round_duration
        self.rest_duration = rest_duration
        self.state = GameState.MENU
        self.stats = SessionStats()

        # Timing
        self.round_start_time = 0.0
        self.last_cue_time = 0.0
        self.countdown_start = 0.0
        self.countdown_duration = 3.0

        # Current cue
        self.active_cue: DefenseCue | None = None

        # Difficulty
        self.cue_min_interval = config.CUE_MIN_INTERVAL
        self.cue_max_interval = config.CUE_MAX_INTERVAL
        self.reaction_window = config.CUE_REACTION_WINDOW

        # Scoring
        self.points_per_punch = 10
        self.points_per_defense = 25
        self.points_per_combo_bonus = 5  # extra per combo level
        self.points_penalty_miss = -10

    def start_round(self):
        """Start a new training round."""
        self.state = GameState.COUNTDOWN
        self.countdown_start = time.time()

    def update(self, punch_class: str | None, defense_class: str | None):
        """
        Called each frame with current predictions.

        Args:
            punch_class: Detected punch type or None.
            defense_class: Detected defense move or None.
        """
        now = time.time()

        if self.state == GameState.COUNTDOWN:
            elapsed = now - self.countdown_start
            if elapsed >= self.countdown_duration:
                self.state = GameState.TRAINING
                self.round_start_time = now
                self.last_cue_time = now
            return

        if self.state != GameState.TRAINING:
            return

        # Check round timer
        elapsed = now - self.round_start_time
        if elapsed >= self.round_duration:
            self.state = GameState.ROUND_END
            return

        # Register punches
        if punch_class and punch_class != "neutral":
            self.stats.punches_detected[punch_class] = (
                self.stats.punches_detected.get(punch_class, 0) + 1
            )
            self.stats.score += self.points_per_punch

        # Handle active cue
        if self.active_cue is not None:
            if self.active_cue.expired and not self.active_cue.responded:
                # Missed the cue
                self.stats.defenses_missed += 1
                self.stats.defenses_total += 1
                self.stats.combo_current = 0
                self.stats.score += self.points_penalty_miss
                self.active_cue = None
            elif defense_class and not self.active_cue.responded:
                # Check if defense matches the cue
                if defense_class == self.active_cue.cue_type:
                    self.active_cue.responded = True
                    self.active_cue.correct = True
                    self.stats.defenses_correct += 1
                    self.stats.defenses_total += 1
                    self.stats.combo_current += 1
                    self.stats.combo_best = max(self.stats.combo_best, self.stats.combo_current)
                    self.stats.score += (
                        self.points_per_defense +
                        self.stats.combo_current * self.points_per_combo_bonus
                    )
                    self.active_cue = None

        # Generate new cue if none active
        if self.active_cue is None:
            time_since_last = now - self.last_cue_time
            next_interval = random.uniform(self.cue_min_interval, self.cue_max_interval)

            if time_since_last >= next_interval:
                self._generate_cue()
                self.last_cue_time = now

    def _generate_cue(self):
        """Generate a random defensive cue."""
        template = random.choice(CUE_TEMPLATES)
        self.active_cue = DefenseCue(
            cue_type=template["type"],
            display_text=template["text"],
            created_at=time.time(),
            reaction_window=self.reaction_window,
        )

    def get_round_time_remaining(self) -> float:
        """Get remaining time in current round (seconds)."""
        if self.state != GameState.TRAINING:
            return self.round_duration
        elapsed = time.time() - self.round_start_time
        return max(0.0, self.round_duration - elapsed)

    def get_countdown_remaining(self) -> float:
        """Get remaining countdown time."""
        if self.state != GameState.COUNTDOWN:
            return 0.0
        elapsed = time.time() - self.countdown_start
        return max(0.0, self.countdown_duration - elapsed)

    def get_cue_time_remaining(self) -> float:
        """Get time remaining to respond to active cue."""
        if self.active_cue is None or self.active_cue.responded:
            return 0.0
        elapsed = time.time() - self.active_cue.created_at
        return max(0.0, self.active_cue.reaction_window - elapsed)

    def next_round(self):
        """Advance to next round."""
        self.stats.round_number += 1
        self.active_cue = None
        self.start_round()

    def reset(self):
        """Reset all state for a new session."""
        self.stats = SessionStats()
        self.state = GameState.MENU
        self.active_cue = None

    def set_difficulty(self, level: str):
        """Adjust difficulty: 'easy', 'medium', 'hard'."""
        settings = {
            "easy": (3.0, 6.0, 3.0),
            "medium": (2.0, 5.0, 2.0),
            "hard": (1.0, 3.0, 1.5),
        }
        if level in settings:
            self.cue_min_interval, self.cue_max_interval, self.reaction_window = settings[level]
