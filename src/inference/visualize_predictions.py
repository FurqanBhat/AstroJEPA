import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from src.masking.collator import IJepaMaskCollator


def visualize_full_latent_map(model, dataset, device='cpu', idx=None):

    model.eval()

    collator = IJepaMaskCollator(
        input_size=model.config['img_size'],
        patch_size=model.config['patch_size'],
        enc_mask_scale=model.config['context_scale'],
        pred_mask_scale=model.config['mask_scale'],
        num_targets=1
    )

    # 1. pick a random galaxy
    if idx==None:
        idx = np.random.randint(0, len(dataset))
    img, _ = dataset[idx]  # img is CHW

    # 2. collate
    images, ctx_mask, tgt_masks = collator([(img, 0)])
    images = images.to(device)
    ctx_mask = ctx_mask.to(device)
    tgt_mask = tgt_masks[0].to(device).bool()   # shape [1, N]

    with torch.no_grad():
        target_emb = model.forward_target(images)[0]           # (N, C)
        context_emb = model.forward_context(images, ctx_mask)[0]
        pred_emb = model.forward_predictor(
            context_emb.unsqueeze(0), 
            [tgt_mask]
        )[0][0]

    N, C = target_emb.shape
    grid = int(np.sqrt(N))

    # reshape masks for visualization
    ctx_mask_img = ctx_mask[0].reshape(grid, grid).cpu().numpy()
    tgt_mask_img = tgt_mask[0].reshape(grid, grid).cpu().numpy()

    # per patch cosine similarity
    pred_norm = F.normalize(pred_emb, dim=-1)
    tgt_norm = F.normalize(target_emb, dim=-1)
    cos = (pred_norm * tgt_norm).sum(-1).cpu().numpy()

    # per patch mse
    mse = ((pred_emb - target_emb)**2).mean(-1).cpu().numpy()

    cos_map = cos.reshape(grid, grid)
    mse_map = mse.reshape(grid, grid)

    # ===============================================
    #   PLOT EVERYTHING INCLUDING MASKS
    # ===============================================
    fig, axes = plt.subplots(1, 5, figsize=(28, 6))

    # original image
    axes[0].imshow(img.permute(1,2,0))
    axes[0].set_title("Original Galaxy")
    axes[0].axis("off")

    # context mask (what JEPA sees)
    axes[1].imshow(ctx_mask_img, cmap='Greens')
    axes[1].set_title("Context Mask (Seen)")
    axes[1].axis("off")

    # target mask (what JEPA predicts)
    axes[2].imshow(tgt_mask_img, cmap='Reds')
    axes[2].set_title("Target Mask (Predicted)")
    axes[2].axis("off")

    # cosine similarity map
    im1 = axes[3].imshow(cos_map, cmap='viridis', vmin=-1, vmax=1)
    axes[3].set_title("Patch Cosine Similarity")
    axes[3].axis("off")
    fig.colorbar(im1, ax=axes[3])

    # patch mse
    im2 = axes[4].imshow(mse_map, cmap='magma')
    axes[4].set_title("Patch MSE")
    axes[4].axis("off")
    fig.colorbar(im2, ax=axes[4])

    plt.tight_layout()
    plt.show()

    print("Random galaxy index:", idx)
    print("Avg Cosine:", cos.mean())
    print("Avg MSE:", mse.mean())
