import copy

import torch
import torch.nn as nn

from layers import (
    ConvLayer,
    FeatureAttentionLayer,
    MultiHeadAttention,
    PositionwiseFeedForward,
    PositionalEncoding,
    TransformerModel,
    TransformerEncoder,
    TransformerEncoderLayer,
)


class Graph_Transformer(nn.Module):
    def __init__(
        self,
        n_features,
        window_size,
        kernel_size=7,
        feat_gat_embed_dim=None,
        use_gatv2=True,
        dropout=0.2,
        alpha=0.2,
        N=2,
        d_ff=0,
        h=8,
    ):
        super().__init__()
        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(
            n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2
        )
        d_model = n_features * 2
        if d_ff == 0:
            d_ff = d_model * 4
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout, window_size)
        # final_linear = nn.Linear(d_model, d_model)
        final_linear = nn.Linear(d_model, n_features)
        self.transformer_encoder = TransformerModel(
            TransformerEncoder(
                TransformerEncoderLayer(d_model, c(attn), c(ff), dropout), N
            ),
            nn.Sequential(position),
            final_linear,
        )

        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, src_mask):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_cat = torch.cat([x, h_feat], dim=2)  # (b, n, 2k)
        output = self.transformer_encoder(h_cat, src_mask)
        return output
