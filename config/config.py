import torch

CONFIG = {
    'batch_size': 256,        # Increased for speed. If OOM, drop to 128.
    'lr': 1e-4,               # Base learning rate
    'warmup_epochs': 10,       # Learning rate warmup
    'weight_decay': 0.05,
    'epochs': 100,             
    'img_size': 224,
    'embed_dim': 192,
    'num_heads': 3,
    'enc_depth': 6,
    'pred_depth': 6,
    'patch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mask_scale': (0.2, 0.3),
    'context_scale': (0.5, 0.7),
    'num_targets': 4,         # Number of target blocks
    'ema_momentum_base': 0.996,  # Will be adjusted based on epochs
    'patience': 10,           # Early stopping patience
    'min_context_patches': 20,  # Minimum visible patches in context
    'center_crop_size': 140,  # Crop size for galaxies
}


def get_config():
    if CONFIG['epochs'] < 100:
        CONFIG['ema_momentum'] = 0.99  # Lower momentum for shorter training
    else:
        CONFIG['ema_momentum'] = CONFIG['ema_momentum_base']

    print(f"Running on: {CONFIG['device']}")
    print(f"Training for {CONFIG['epochs']} epochs with EMA momentum: {CONFIG['ema_momentum']}")

    return CONFIG
