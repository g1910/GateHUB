import torch
import torch.nn as nn

from .attention import AttentionLayer, FullAttention, GatedAttention
from .blocks import (
    DecoderLayer,
    EncoderLayer,
    GatedHistoryUnit,
    HistoryEncoder,
    PresentDecoder,
)
from .position_encoding import FixedPositionalEncoding, LearnedPositionalEncoding

# Maps parameter prefixes from the original research checkpoints onto the
# current module names; see GateHUB.remap_legacy_state_dict.
_LEGACY_KEY_MAP = {
    'memory_decoder_cls_token_history': 'latent_query',
    'short_position_encoding': 'present_position_encoding',
    'history_encoder.decoder': 'history_encoder.ghu',
    'mlp_head': 'classifier',
}


class GateHUB(nn.Module):
    """Gated History Unit with Background suppression for online action detection.

    Encodes the observed history into a fixed-size latent via position-guided
    gated cross-attention (GHU), then correlates it with the present to make the
    C+1-way prediction for the current frame.
    """

    def __init__(self, args):
        super(GateHUB, self).__init__()

        self.args = args
        self.use_rgb = not args.flow_only
        self.use_flow = not args.rgb_only

        d_model = args.embedding_dim
        d_ff = args.decoder_embedding_dim_out
        n_heads = args.decoder_num_heads
        dropout = args.decoder_attn_dropout_rate
        activation = 'gelu'

        self.linear_encoding = nn.Linear(args.dim_feature, d_model)
        self.gate_layer = nn.Linear(d_model, 1)
        self.after_dropout = nn.Dropout(p=args.dropout_rate)

        if args.positional_encoding_type == "learned":
            self.history_position_encoding = LearnedPositionalEncoding(
                args.history_frames, d_model, args.history_frames)
            self.present_position_encoding = LearnedPositionalEncoding(
                args.present_frames, d_model, args.present_frames)
        else:
            self.history_position_encoding = FixedPositionalEncoding(d_model)
            self.present_position_encoding = FixedPositionalEncoding(d_model)

        # Latent encoding q (L x D) that the history is cross-attended into.
        init = torch.randn if args.dec_init == "random" else torch.zeros
        self.latent_query = nn.Parameter(init(1, args.latent_size, d_ff))

        self.history_encoder = HistoryEncoder(
            GatedHistoryUnit(
                AttentionLayer(GatedAttention(attention_dropout=dropout), d_ff, n_heads),
                d_ff, d_ff, dropout=dropout, activation=activation,
            ),
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_ff, n_heads),
                    d_ff, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(args.history_layers)
            ],
            norm_layer=nn.LayerNorm(d_ff),
        )

        self.present_decoder = PresentDecoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, attention_dropout=dropout), d_ff, n_heads),
                    AttentionLayer(FullAttention(False, attention_dropout=dropout), d_ff, n_heads),
                    d_ff, d_ff, dropout=dropout, activation=activation,
                )
                for _ in range(args.decoder_layers)
            ],
            norm_layer=nn.LayerNorm(d_ff),
        )

        self.classifier = nn.Linear(d_ff, args.numclass)

    @staticmethod
    def remap_legacy_state_dict(state_dict):
        """Rename parameters saved by the original research code to current names."""
        remapped = {}
        for key, value in state_dict.items():
            for old, new in _LEGACY_KEY_MAP.items():
                if key == old or key.startswith(old + '.'):
                    key = new + key[len(old):]
                    break
            remapped[key] = value
        return remapped

    def _fuse(self, rgb, flow):
        if self.use_rgb and self.use_flow:
            return torch.cat((rgb, flow), dim=2)
        return rgb if self.use_rgb else flow

    def forward(self, present_rgb, present_flow, history_rgb, history_flow):
        present_feat = self.linear_encoding(self._fuse(present_rgb, present_flow))
        present_feat = self.present_position_encoding(present_feat)

        history_feat = self.linear_encoding(self._fuse(history_rgb, history_flow))
        history_feat = self.history_position_encoding(history_feat)

        # Eqn. 1-2: z = sigmoid(z_h W_g); G = log(z) + z. The epsilon keeps log finite.
        gate_scores = torch.sigmoid(self.gate_layer(history_feat)).squeeze(-1) + 1e-8

        latent = self.latent_query.expand(present_feat.shape[0], -1, -1)
        history_embed = self.history_encoder(latent, history_feat, gate_scores=gate_scores)
        history_embed = self.after_dropout(history_embed)

        out = self.present_decoder(present_feat, history_embed)
        out = self.after_dropout(out)

        return self.classifier(out)
