import torch
import torch.nn as nn
from src.models.patch_embed import PatchEmbed
from src.models.vit import MiniViT


class MiniJEPA(nn.Module):
    """Mini JEPA Model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Shared patch embedding
        self.patch_embed = PatchEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            embed_dim=config['embed_dim']
        )

        num_patches = (config['img_size'] // config['patch_size']) ** 2
        
        # 2. Context and target encoders
        self.context_encoder = MiniViT(
            num_patches=num_patches,
            embed_dim=config['embed_dim'],
            depth=config['enc_depth'],
            num_heads=config['num_heads']
        )
        self.target_encoder = MiniViT(
            num_patches=num_patches,
            embed_dim=config['embed_dim'],
            depth=config['enc_depth'],
            num_heads=config['num_heads']
        )

        # 3. Predictor
        self.predictor = MiniViT(
            num_patches=num_patches,
            embed_dim=config['embed_dim'],
            depth=config['pred_depth'],
            num_heads=config['num_heads']
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim']))

        self._init_weights()

        # Copy context encoder weights into target encoder and freeze
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_target(self, images):
        """Forward pass through target encoder"""
        x = self.patch_embed(images)
        x = x + self.target_encoder.pos_embed
        x = self.target_encoder(x)
        return x

    def forward_context(self, images, context_mask):
        """Forward pass through context encoder"""
        x = self.patch_embed(images)
        x = x + self.context_encoder.pos_embed

        # PyTorch Transformer uses True = ignore, so invert
        padding_mask = ~context_mask.bool()

        x = self.context_encoder(x, padding_mask=padding_mask)
        return x

    def forward_predictor(self, context_output, target_masks):
        """Predict target representations from context"""
        B, N, C = context_output.shape
        predictions = []

        pos_embed_expanded = self.predictor.pos_embed.expand(B, N, C)

        for t_mask in target_masks:
            t_mask = t_mask.bool()

            pred_input = context_output.clone()
            pred_input = pred_input + pos_embed_expanded

            mask_tokens = self.mask_token.expand(B, N, C)
            pred_input[t_mask] = mask_tokens[t_mask] + pos_embed_expanded[t_mask]

            out = self.predictor(pred_input)
            predictions.append(out)

        return predictions
