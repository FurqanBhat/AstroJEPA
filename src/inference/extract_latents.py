import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def extract_latents(model, dataset, device='cuda', batch_size=64):
    """
    Extracts latents from a model and dataset.
    """
    # specific check to avoid cuda errors if user forgot to check availability
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latents = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting Latents"):
            imgs = imgs.to(device)
            
            # Assuming model.forward_target exists as per your snippet
            z = model.forward_target(imgs)   # (B, N, C)
            z = z.mean(dim=1)                # pooled representation (B, C)
            
            latents.append(z.cpu().numpy())
            labels.append(lbls.numpy())

    latents = np.concatenate(latents, axis=0)
    labels = np.concatenate(labels, axis=0)

    # CRITICAL FIX: Cast to float32 HERE, before returning. 
    # This prevents memory bloat and ensures compatibility with UMAP immediately.
    latents = latents.astype(np.float32)

    return latents, labels