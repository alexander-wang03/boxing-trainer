"""
Model A: Bidirectional LSTM Punch Classifier.

Input:  (batch, seq_len, features_per_frame)
        BoxingVI: (batch, 25, 34) — 25 frames, 17 COCO keypoints x 2 coords
        Custom:   (batch, 30, 99) — 30 frames, 33 MediaPipe keypoints x 3 coords
Output: (batch, 9) — logits for 8 punch types + neutral

Architecture:
    1. Per-frame FC: features -> 128, ReLU, Dropout(0.2)
    2. 2-layer Bidirectional LSTM, hidden_size=256
    3. Classifier: 512 -> 128 -> 9
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config


class PunchClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = config.FEATURES_PER_FRAME,
        fc_dim: int = config.PUNCH_FC_DIM,
        lstm_hidden: int = config.PUNCH_LSTM_HIDDEN,
        lstm_layers: int = config.PUNCH_LSTM_LAYERS,
        num_classes: int = config.NUM_PUNCH_CLASSES,
        dropout: float = config.PUNCH_DROPOUT,
    ):
        super().__init__()

        # Per-frame feature projection
        self.frame_fc = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=fc_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Classifier head (bidirectional → hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features_per_frame)

        Returns:
            logits: (batch, num_classes)
        """
        # Per-frame projection
        batch, _, _ = x.shape

        # Find valid (non-padded) frame counts per sequence
        valid_mask = (x.abs().sum(dim=-1) > 1e-6)  # (batch, seq_len)
        seq_lengths = valid_mask.long().sum(dim=1).clamp(min=1)  # (batch,)

        x = self.frame_fc(x)  # (batch, seq_len, fc_dim)

        # Pack to skip padded frames — fixes BiLSTM backward direction
        packed = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, max_len, hidden*2)

        # Gather output at last valid frame for each sequence
        idx = (seq_lengths - 1).clamp(min=0).view(batch, 1, 1)
        idx = idx.expand(batch, 1, lstm_out.size(2))
        final_output = lstm_out.gather(1, idx).squeeze(1)  # (batch, lstm_hidden * 2)

        # Classify
        logits = self.classifier(final_output)  # (batch, num_classes)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = PunchClassifier()
    print(f"PunchClassifier: {count_parameters(model):,} parameters")

    # Test forward pass
    dummy = torch.randn(4, config.SEQUENCE_LENGTH, config.FEATURES_PER_FRAME)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
