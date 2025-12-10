from config.config import get_config
import torchvision.transforms as T
from src.data.dataset import Galaxy10Dataset, TransformDataset
import torch
from torch.utils.data import DataLoader, random_split
from src.masking.collator import IJepaMaskCollator
import os
from PIL import Image
from torch.utils.data import Dataset



data_path='Galaxy10.h5'

full_dataset = Galaxy10Dataset(file_path=data_path, transform=None)


CONFIG = get_config()


# Training augmentations (strong augmentation to prevent collapse)
train_transform = T.Compose([
    T.CenterCrop(CONFIG['center_crop_size']),  # Zoom in on galaxy
    T.RandomRotation(180),      # Galaxies rotate!
    T.RandomHorizontalFlip(),   # Chirality invariance
    T.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Test transform (no random augmentation!)
test_transform = T.Compose([
    T.CenterCrop(CONFIG['center_crop_size']),
    T.Resize((CONFIG['img_size'], CONFIG['img_size'])),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("=" * 60)
print("Initializing Dataset")
print("=" * 60)


# full_dataset = Galaxy10Dataset(file_path=data_path, transform=None)


# Split dataset
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_subset, test_subset = random_split(
    full_dataset, 
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Reproducible split
)

# Apply transforms
train_subset = TransformDataset(train_subset, train_transform)
test_subset = TransformDataset(test_subset, test_transform)

# Create collator
collator = IJepaMaskCollator(
    input_size=CONFIG['img_size'],
    patch_size=CONFIG['patch_size'],
    enc_mask_scale=CONFIG['context_scale'],
    pred_mask_scale=CONFIG['mask_scale'],
    num_targets=CONFIG['num_targets'],
    min_context_patches=CONFIG['min_context_patches']
)

# DataLoaders (num_workers=0 since data is in RAM)
train_loader = DataLoader(
    train_subset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True,
    collate_fn=collator, 
    drop_last=True, 
    num_workers=2,  # 0 for in-RAM dataset
    pin_memory=True
)
test_loader = DataLoader(
    test_subset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    collate_fn=collator, 
    drop_last=False, 
    num_workers=3,
    pin_memory=True
)

print(f"✓ Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
print(f"✓ Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")


def get_data():
    return train_loader, test_loader

def get_test_subset():
    return test_subset

    
