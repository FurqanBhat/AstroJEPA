import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.masking.collator import IJepaMaskCollator




@torch.no_grad()
def compute_jepa_anomaly_scores(
    model,
    dataset,
    batch_size=32,
    device="cpu",
    save_path=None
):
    """
    Compute per-galaxy anomaly scores using JEPA latent prediction error.

    For each image:
      - use IJepaMaskCollator to sample context & target blocks
      - run target encoder, context encoder, predictor
      - compute cosine similarity and MSE on MASKED patches only
      - aggregate:
          mean_mse, max_mse, mean_cos, min_cos, num_masked_patches
      - combine into a scalar anomaly_score

    Returns:
      df: pandas DataFrame with one row per image in `dataset`.
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # masking config from model
    img_size = model.config["img_size"]
    patch_size = model.config["patch_size"]
    context_scale = model.config["context_scale"]
    mask_scale = model.config["mask_scale"]
    num_targets = model.config["num_targets"]
    min_ctx = model.config.get("min_context_patches", 8)

    collator = IJepaMaskCollator(
        input_size=img_size,
        patch_size=patch_size,
        enc_mask_scale=context_scale,
        pred_mask_scale=mask_scale,
        num_targets=num_targets,
        min_context_patches=min_ctx,
    )

    num_samples = len(dataset)

    rows = {
        "index": [],
        "label": [],
        "mean_mse": [],
        "max_mse": [],
        "mean_cos": [],
        "min_cos": [],
        "num_masked_patches": [],
        "anomaly_score": [],
    }

    print(f"Computing anomaly scores on {num_samples} images using {device}.")

    for start in tqdm(range(0, num_samples, batch_size)):
        end = min(start + batch_size, num_samples)

        # manual mini-batch from dataset
        batch_samples = [dataset[i] for i in range(start, end)]
        imgs = [x[0] for x in batch_samples]
        labels = torch.tensor([int(x[1]) for x in batch_samples])

        images, ctx_mask, tgt_masks = collator(
            [(img, lbl) for img, lbl in zip(imgs, labels)]
        )

        images = images.to(device)
        ctx_mask = ctx_mask.to(device)
        tgt_masks = [m.to(device).bool() for m in tgt_masks]

        # forward passes
        target_emb = model.forward_target(images)              # (B, N, C)
        context_emb = model.forward_context(images, ctx_mask)  # (B, N, C)
        preds_list = model.forward_predictor(context_emb, tgt_masks)
        # preds_list: list of length num_targets, each (B, N, C)

        B, N, C = target_emb.shape

        # collect per image
        per_img_cos = [[] for _ in range(B)]
        per_img_mse = [[] for _ in range(B)]

        for pred_block, mask_block in zip(preds_list, tgt_masks):
            # pred_block: (B, N, C), mask_block: (B, N)
            for b in range(B):
                m = mask_block[b]      # (N,)
                if not m.any():
                    continue

                p = pred_block[b][m]   # (#patch, C)
                t = target_emb[b][m]   # (#patch, C)

                # cosine similarity per patch
                cos_vals = F.cosine_similarity(p, t, dim=-1)  # (#patch,)
                # mse per patch (over embedding dim)
                mse_vals = F.mse_loss(p, t, reduction="none").mean(dim=-1)

                per_img_cos[b].append(cos_vals.cpu())
                per_img_mse[b].append(mse_vals.cpu())

        # aggregate stats and anomaly score
        for local_idx in range(B):
            global_idx = start + local_idx
            label = int(labels[local_idx].item())

            if len(per_img_mse[local_idx]) == 0:
                # nothing masked, should not happen but whatever
                mean_mse = float("nan")
                max_mse = float("nan")
                mean_cos = float("nan")
                min_cos = float("nan")
                num_masked = 0
                anomaly = float("nan")
            else:
                mse_all = torch.cat(per_img_mse[local_idx])    # (K,)
                cos_all = torch.cat(per_img_cos[local_idx])    # (K,)

                mean_mse = mse_all.mean().item()
                max_mse = mse_all.max().item()
                mean_cos = cos_all.mean().item()
                min_cos = cos_all.min().item()
                num_masked = mse_all.numel()

                # simple hand tuned anomaly score:
                #   base on mean error, penalize large peaks, reward low cosine
                anomaly = (
                    mean_mse
                    + 0.5 * max_mse
                    + 0.1 * (1.0 - mean_cos)
                )

            rows["index"].append(global_idx)
            rows["label"].append(label)
            rows["mean_mse"].append(mean_mse)
            rows["max_mse"].append(max_mse)
            rows["mean_cos"].append(mean_cos)
            rows["min_cos"].append(min_cos)
            rows["num_masked_patches"].append(num_masked)
            rows["anomaly_score"].append(anomaly)

    df = pd.DataFrame(rows)
    df = df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)

    if save_path is not None:
        df.to_csv(save_path, index=False)
        print(f"Saved anomaly scores to {save_path}")

    return df