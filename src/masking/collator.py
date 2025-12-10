import torch
import numpy as np


class IJepaMaskCollator:
    """
    Galaxy-aware masking collator for Mini-JEPA on Galaxy10.

    Returns:
        images:        (B, C, H, W)
        context_mask:  (B, N) bool, True = visible to context encoder
        target_masks:  list of length num_targets, each (B, N) bool, True = target block
    """

    def __init__(
        self,
        input_size=224,
        patch_size=32,
        enc_mask_scale=(0.4, 0.8),   # size of VISIBLE context block, in fraction of grid dims
        pred_mask_scale=(0.2, 0.3),  # size of each TARGET block, in fraction of grid dims
        num_targets=4,
        min_context_patches=10
    ):
        self.input_size = input_size
        self.patch_size = patch_size

        self.H = input_size // patch_size   # patch grid height
        self.W = input_size // patch_size   # patch grid width
        self.N = self.H * self.W

        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.num_targets = num_targets
        self.min_context_patches = min_context_patches

    # ------------------------------------------------------
    def __call__(self, batch):
        """
        batch: list of (image, label) or (image,)
        """
        # Extract images only
        images = torch.stack([item[0] for item in batch])  # (B, C, H, W)
        B = images.shape[0]

        # 1. Compute brightness per patch (so we know where the galaxy actually is)
        brightness = self._compute_patch_brightness(images)  # (B, H, W)

        # 2. Sample target masks (what the predictor must reconstruct)
        target_masks = []
        for _ in range(self.num_targets):
            t_mask = self._sample_semantic_block(brightness, self.pred_mask_scale)  # (B, N)
            target_masks.append(t_mask)

        # Combine all target blocks
        full_target_mask = torch.stack(target_masks, dim=0).any(dim=0)  # (B, N)

        # 3. Sample context mask (what the context encoder sees)
        #    This block represents the VISIBLE region.
        context_mask = self._sample_semantic_block(brightness, self.enc_mask_scale)  # (B, N)

        # Remove any overlap with target blocks
        context_mask = context_mask & (~full_target_mask)

        # 4. Ensure enough context patches remain
        for b in range(B):
            if context_mask[b].sum() < self.min_context_patches:
                # Fallback: allow every non-target patch to be context
                context_mask[b] = ~full_target_mask[b]
                # Still possible this is all False if target covers everything,
                # but with reasonable scales that should never happen.

        return images, context_mask, target_masks

    # ------------------------------------------------------
    def _compute_patch_brightness(self, imgs):
        """
        Compute mean absolute brightness per patch.

        imgs: (B, C, H, W), normalized, can be in [-1,1] or similar.
        Returns: (B, H_patches, W_patches)
        """
        B, C, H, W = imgs.shape
        assert H == self.input_size and W == self.input_size, \
            f"Expected images of size {self.input_size}x{self.input_size}, got {H}x{W}"

        # Approx "grayscale" intensity
        gray = imgs.mean(dim=1).abs()  # (B, H, W)

        # Fold into patches of size patch_size x patch_size
        p = self.patch_size
        Hp = H // p
        Wp = W // p

        # unfold: (B, Hp, Wp, p, p)
        patches = gray.unfold(1, p, p).unfold(2, p, p)
        # Mean brightness per patch
        brightness = patches.mean(dim=(-1, -2))  # (B, Hp, Wp)

        return brightness

    # ------------------------------------------------------
    def _sample_semantic_block(self, brightness, scale):
        """
        Sample a block of patches per image, biased toward bright (galaxy) regions.

        brightness: (B, H, W)
        scale: (min_frac, max_frac) of grid dims for block size
        Returns:
            mask: (B, N) bool, True where block is active (visible or target)
        """
        B, H, W = brightness.shape
        mask = torch.zeros(B, H, W, dtype=torch.bool)

        min_s, max_s = scale

        for b in range(B):
            br = brightness[b]  # (H, W)

            # pick block size as a fraction of grid dims
            block_h = np.random.randint(
                max(1, int(H * min_s)),
                max(1, int(H * max_s)) + 1
            )
            block_w = np.random.randint(
                max(1, int(W * min_s)),
                max(1, int(W * max_s)) + 1
            )

            # probability from brightness
            prob = br.flatten()  # (H*W,)

            # sanitize probs
            prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
            prob = torch.clamp(prob, min=0.0)

            if prob.sum() <= 0:
                # if galaxy is very faint, fall back to uniform
                prob = torch.ones_like(prob)

            prob = prob / prob.sum()

            # sample center location
            idx = torch.multinomial(prob, 1).item()
            cy = idx // W
            cx = idx % W

            # convert center → top-left
            y0 = int(max(0, cy - block_h // 2))
            x0 = int(max(0, cx - block_w // 2))

            # clamp to stay inside grid
            y0 = min(y0, H - block_h)
            x0 = min(x0, W - block_w)

            y1 = y0 + block_h
            x1 = x0 + block_w

            mask[b, y0:y1, x0:x1] = True

        return mask.view(B, H * W)
