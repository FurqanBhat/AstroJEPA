import torch
import torch.nn as nn

class MiniViT(nn.Module):
    """Mini Vision Transformer"""
    def __init__(self, num_patches=196, embed_dim=192, depth=3, num_heads=3):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, padding_mask=None):
        # x: (B, N, C)
        # padding_mask: (B, N) where True means "Ignore this token"
        x = self.blocks(x, src_key_padding_mask=padding_mask)
        return self.norm(x)