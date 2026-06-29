"""The GPT model: a decoder-only transformer language model.

Architecture (GPT-2 family):

    tokens -> token embedding + learned positional embedding
           -> N x pre-norm transformer blocks
           -> final LayerNorm
           -> linear head (weight-tied to the token embedding)
           -> logits over the vocabulary
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from slm.config import ModelConfig
from slm.model.layers import Block, LayerNorm

__all__ = ["GPT"]


class GPT(nn.Module):
    """A configurable GPT-style causal language model.

    Submodules are exposed as explicit, typed attributes (rather than a string
    keyed ``ModuleDict``) so static type checkers can verify every access.

    Args:
        config: Architecture hyper-parameters.
        gradient_checkpointing: Trade compute for memory by recomputing block
            activations during the backward pass.
    """

    def __init__(self, config: ModelConfig, *, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: the input embedding and output projection share weights,
        # which improves quality and removes ~vocab*n_embd parameters.
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Apply a scaled init to residual projections (GPT-2 §2.3).
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # -- initialisation -----------------------------------------------------
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, *, non_embedding: bool = True) -> int:
        """Total parameter count. Positional embeddings are excluded by default.

        The token embedding is tied to the head, so it is counted once (as part
        of the head) and not double-counted here.
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.position_embedding.weight.numel()
        return n

    # -- forward ------------------------------------------------------------
    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run a forward pass.

        Args:
            idx: ``(B, T)`` int64 tensor of token ids, ``T <= block_size``.
            targets: Optional ``(B, T)`` int64 tensor of next-token labels. When
                provided, the cross-entropy loss is returned and logits are
                computed for every position.

        Returns:
            ``(logits, loss)``. When ``targets`` is ``None`` only the final
            position's logits are computed (an inference optimisation) and
            ``loss`` is ``None``.
        """
        device = idx.device
        _, t = idx.size()
        if t > self.config.block_size:
            raise ValueError(f"Sequence length {t} exceeds block_size {self.config.block_size}.")
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss

        # Inference: only the last position is needed to predict the next token.
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # -- optimiser ----------------------------------------------------------
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """Build an AdamW optimiser with sensible weight-decay grouping.

        Tensors with >= 2 dimensions (matmul/embedding weights) are decayed;
        biases and LayerNorm gains are not. The fused AdamW kernel is used on
        CUDA when available.
        """
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda" and "fused" in _adamw_signature()
        extra = {"fused": True} if use_fused else {}  # pragma: no cover - fused is CUDA-only
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)

    # -- introspection ------------------------------------------------------
    def parameter_groups_numel(self) -> dict[str, int]:
        """Return a small breakdown of parameter counts for logging."""
        return {
            "embedding": self.token_embedding.weight.numel(),
            "position": self.position_embedding.weight.numel(),
            "total": self.get_num_params(non_embedding=False),
            "non_embedding": self.get_num_params(non_embedding=True),
        }


def _adamw_signature() -> Iterable[str]:
    import inspect

    return inspect.signature(torch.optim.AdamW).parameters.keys()
