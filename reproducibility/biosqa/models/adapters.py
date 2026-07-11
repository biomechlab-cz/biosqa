"""Per-modality input adapters (Plan 1 §7.2).

Each modality has its own PatchTST-style patch tokenizer: a strided 1D conv that
turns a raw window ``[B, C_in, L]`` into a token sequence ``[B, n_tokens,
d_model]``. Differing sampling rates are absorbed by choosing ``patch_len`` /
``stride`` per modality (see :mod:`biosqa.data.windows`) so token counts are
comparable across modalities (~32-64 tokens/window). A learned positional
embedding and a shared modality-type embedding are added so a single backbone
can disambiguate modalities (PanLUNA/PhysioOmni pattern).

Everything here is plain conv/linear/embedding — it traces cleanly to ONNX.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PatchAdapter", "n_tokens_for"]


def n_tokens_for(length: int, patch_len: int, stride: int) -> int:
    """Number of patch tokens a Conv1d(kernel=patch_len, stride) produces."""
    return (length - patch_len) // stride + 1


class PatchAdapter(nn.Module):
    """Conv1d patch-embed tokenizer for one modality.

    Parameters
    ----------
    patch_len, stride : patch tokenizer geometry (pick per modality so token
        counts match across modalities).
    d_model : token/embedding width (shared backbone width).
    c_in : input channels (1 for the deployment contract ``[1, 1, L]``; >1 for
        multi-lead ECG / multi-channel EEG when trained multivariately).
    max_tokens : size of the learned positional embedding table.
    """

    def __init__(
        self,
        patch_len: int,
        stride: int,
        d_model: int = 128,
        c_in: int = 1,
        max_tokens: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.c_in = c_in
        self.proj = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=stride)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C_in, L] -> [B, n_tokens, d_model]`` with positional embedding."""
        if x.dim() == 2:  # [B, L] -> [B, 1, L]
            x = x.unsqueeze(1)
        z = self.proj(x)                 # [B, d_model, n_tokens]
        z = z.transpose(1, 2)            # [B, n_tokens, d_model]
        n = z.size(1)
        z = z + self.pos_embed[:, :n, :]
        return self.dropout(z)
