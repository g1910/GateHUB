import numpy as np
import torch
import torch.nn as nn

from math import sqrt


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    """Scaled dot-product attention, optionally with a causal mask."""

    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, gate_scores=None):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)
            else:
                scores.masked_fill_(
                    attn_mask.bool().unsqueeze(1).unsqueeze(2).expand(
                        -1, scores.shape[1], scores.shape[2], -1),
                    -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class GatedAttention(nn.Module):
    """Gated cross-attention of the Gated History Unit (Eqn. 3).

    Adds the gating score sequence G to the pre-softmax logits so that each
    history frame's attention weight is rescaled by a factor in [0, e]:
    softmax(QK^T/sqrt(d) + G) with G = log(z) + z, z = sigmoid(...) in [0, 1].

    The division by ``scale`` compensates for the multiplication applied to the
    whole logit tensor below, so G enters the softmax unscaled.
    """

    def __init__(self, scale=None, attention_dropout=0.1):
        super(GatedAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, gate_scores=None):
        _, _, _, E = queries.shape
        scale = self.scale or 1. / sqrt(E)

        z = gate_scores.float().unsqueeze(1).unsqueeze(2)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = scores + (torch.log(z) / scale) + (z / scale)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    """Multi-head projection wrapper around an attention module (Eqn. 4)."""

    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None, gate_scores=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(queries, keys, values, attn_mask, gate_scores).view(B, L, -1)

        return self.out_projection(out)
