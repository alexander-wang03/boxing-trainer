"""
Model B: LSTM Defense Classifier.

Input:  (batch, 30, head_features) — 30-frame sequences of head keypoints + velocity
Output: (batch, 5) — logits for 4 defense types + neutral

Architecture:
    1. 2-layer LSTM, hidden_size=128
    2. Classifier: 128 → 64 → 5
"""

import torch
import torch.nn as nn

import config


# Head features: 11 keypoints × 3 coords = 33 positions + 33 velocities = 66
HEAD_FEATURE_DIM = len(config.HEAD_KEYPOINT_INDICES) * config.KEYPOINT_DIMS * 2


class DefenseClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = HEAD_FEATURE_DIM,
        lstm_hidden: int = config.DEFENSE_LSTM_HIDDEN,
        lstm_layers: int = config.DEFENSE_LSTM_LAYERS,
        fc_dim: int = config.DEFENSE_FC_DIM,
        num_classes: int = config.NUM_DEFENSE_CLASSES,
        dropout: float = config.DEFENSE_DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, head_features)

        Returns:
            logits: (batch, num_classes)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)

        # Use final time step
        final_output = lstm_out[:, -1, :]  # (batch, lstm_hidden)

        logits = self.classifier(final_output)  # (batch, num_classes)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = DefenseClassifier()
    print(f"DefenseClassifier: {count_parameters(model):,} parameters")

    dummy = torch.randn(4, config.SEQUENCE_LENGTH, HEAD_FEATURE_DIM)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
