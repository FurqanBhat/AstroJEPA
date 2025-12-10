import torch
from src.models.jepa import MiniJEPA


def load_jepa_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']

    model = MiniJEPA(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print("Loaded:", path)
    print("Epoch:", ckpt.get('epoch'))
    print("Best Val Loss:", ckpt.get('val_loss'))
    print("Trainable Params:", model.count_parameters())

    return model, config
