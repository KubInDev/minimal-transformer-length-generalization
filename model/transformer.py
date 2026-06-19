"""
model/transformer.py

Key design decisions (vs v1):
  - CLS token prepended to every sequence. Classification reads from CLS only,
    forcing information to flow through attention rather than exploiting the
    positional proximity of the last token to the answer.
  - nhead=2 (even) - avoids PyTorch nested-tensor UserWarning.
  - pos_encoding in {"learned", "sin", "none"}.
    "none" is the ablation baseline: tests whether the model can solve tasks
    without any positional signal (expected to fail on First-Last).
  - dropout=0.0 intentionally - we study raw inductive bias, not regularised
    behaviour.
"""

import numpy as np
import torch
import torch.nn as nn
import config


class MinimalTransformer(nn.Module):

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        nhead: int = config.NHEAD,
        dim_feedforward: int = config.DIM_FF,
        max_len: int = config.MAX_LEN,
        pos_encoding: str = "learned",
    ):
        super().__init__()

        assert pos_encoding in {"learned", "sin", "none"}, (
            f"pos_encoding must be 'learned', 'sin', or 'none', got '{pos_encoding}'"
        )
        self.pos_encoding_type = pos_encoding
        self.d_model = d_model

        # Token embedding: vocab size = 2 (binary sequences)
        self.embedding = nn.Embedding(2, d_model)

        # CLS token - shared learnable vector prepended to every sequence..
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encodings
        if pos_encoding == "learned":
            # +1 because CLS occupies position 0
            self.pos_embedding = nn.Parameter(
                torch.randn(max_len + 1, d_model) * 0.02
            )
        elif pos_encoding == "sin":
            self.register_buffer(
                "pos_embedding",
                self._build_sinusoidal(max_len + 1, d_model),
            )
        # "none": no positional attribute created

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=config.DROPOUT,
            norm_first=False, # post-LN (standard)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(d_model, 2)

    # Sinusoidal PE

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(np.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) # [1, max_len, d_model]

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len]  integer token ids in {0, 1}
        Returns:
            logits: [B, 2]
        """
        B, L = x.shape

        # Token embeddings
        tok = self.embedding(x) # [B, L, d_model]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1) # [B, 1, d_model]
        tok = torch.cat([cls, tok], dim=1) # [B, L+1, d_model]

        # Add positional encoding
        if self.pos_encoding_type == "learned":
            tok = tok + self.pos_embedding[: L + 1]
        elif self.pos_encoding_type == "sin":
            tok = tok + self.pos_embedding[:, : L + 1, :]
        # "none": no addition

        # Single-layer transformer encoder
        h = self.encoder(tok) # [B, L+1, d_model]

        # Read from CLS position only
        cls_repr = h[:, 0, :] # [B, d_model]

        return self.classifier(cls_repr) # [B, 2]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Sanity check
if __name__ == "__main__":
    for pe in ("learned", "sin", "none"):
        m = MinimalTransformer(pos_encoding=pe)
        x = torch.randint(0, 2, (8, 50))
        out = m(x)
        assert out.shape == (8, 2), f"Output shape error for pe={pe}"
        print(f"  pe={pe:<8}  params={m.count_parameters():,}  out={tuple(out.shape)}  ✓")
