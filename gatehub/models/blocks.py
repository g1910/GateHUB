import torch.nn as nn
import torch.nn.functional as F

# Layer/parameter names are kept flat (conv1/conv2/norm1..norm3) so checkpoints
# trained with the original research code load without key remapping.


class DecoderLayer(nn.Module):
    """Self-attention followed by cross-attention (Present Decoder, Fig. 2c)."""

    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, gate_scores=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross, attn_mask=cross_mask, gate_scores=gate_scores))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class GatedHistoryUnit(nn.Module):
    """Gated cross-attention of the latent query against the history (Fig. 2a)."""

    def __init__(self, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(GatedHistoryUnit, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, cross_mask=None, gate_scores=None):
        x = x + self.dropout(self.cross_attention(
            x, cross, cross, attn_mask=cross_mask, gate_scores=gate_scores))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class EncoderLayer(nn.Module):
    """Self-attention block used after the GHU in the History Encoder."""

    def __init__(self, self_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask))

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class PresentDecoder(nn.Module):
    """Stack of DecoderLayers correlating the present with the history encoding."""

    def __init__(self, layers, norm_layer=None):
        super(PresentDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, gate_scores=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, gate_scores=gate_scores)

        if self.norm is not None:
            x = self.norm(x)

        return x


class HistoryEncoder(nn.Module):
    """GHU followed by N self-attention layers (Fig. 2b)."""

    def __init__(self, gated_history_unit, encoder_layers, norm_layer=None):
        super(HistoryEncoder, self).__init__()
        self.ghu = gated_history_unit
        self.layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, gate_scores=None):
        x = self.ghu(x, cross, gate_scores=gate_scores)
        for layer in self.layers:
            x = layer(x, x_mask=x_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
