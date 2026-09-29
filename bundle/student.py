"""Student language model with parameter-matched SwiGLU and LayerNorm.

The default ``mlp`` setting is ``swiglu``. Set ``mlp`` to ``gelu`` in a
configuration for the mechanism ablation while keeping the rest of the
student architecture and training recipe unchanged. Each Transformer block and
the final normalization use LayerNorm.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward network with roughly the baseline MLP parameter count."""

    def __init__(self, width):
        super().__init__()
        # Three projections keep the parameter count close to the baseline MLP.
        inner = max(1, round(8 * width / 3))
        self.value = nn.Linear(width, inner)
        self.gate = nn.Linear(width, inner)
        self.output = nn.Linear(inner, width)

    def forward(self, x):
        return self.output(self.value(x) * F.silu(self.gate(x)))


class StudentBlock(nn.Module):
    def __init__(self, width, heads, mlp_type="swiglu"):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        # Keep both MLP branches available for controlled architecture ablations.
        if mlp_type == "swiglu":
            self.mlp = SwiGLU(width)
        elif mlp_type == "gelu":
            # Student-side ablation: restore the baseline feed-forward network.
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width),
                nn.GELU(),
                nn.Linear(4 * width, width),
            )
        else:
            raise ValueError(f"Unknown mlp type: {mlp_type}")

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class StudentGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config["context"]
        width = config["width"]
        mlp_type = config.get("mlp", "swiglu")

        self.token = nn.Embedding(config["vocab"], width)
        self.pos = nn.Embedding(self.context, width)
        self.blocks = nn.ModuleList([
            StudentBlock(width, config["heads"], mlp_type)
            for _ in range(config["depth"])
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config["vocab"], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.pos(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, ids):
        """Return unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Return finite, normalized causal log probabilities."""
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentGPT(config)
