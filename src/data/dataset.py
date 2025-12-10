"""Dataset classes for Galaxy10."""

import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Callable, Tuple
import torch
from torch.utils.data import Dataset


class Galaxy10Dataset(Dataset):
    """
    Galaxy10 dataset with flexible loading strategy.
    
    Args:
        file_path: Path to Galaxy10_DECals.h5 file
        transform: Optional transform to apply to images
        load_to_ram: If True, load entire dataset to RAM (default: False for local use)
                    Set to True only if you have 8GB+ free RAM and want fastest training
    
    Usage:
        # For inference/small experiments (lazy loading, memory efficient)
        dataset = Galaxy10Dataset('data/Galaxy10_DECals.h5', load_to_ram=False)
        
        # For full training (faster but uses ~6GB RAM)
        dataset = Galaxy10Dataset('data/Galaxy10_DECals.h5', load_to_ram=True)
    """
    
    def __init__(
        self,
        file_path: str,
        transform: Optional[Callable] = None,
        load_to_ram: bool = False  # Changed default to False
    ):
        self.transform = transform
        self.file_path = Path(file_path)
        self.load_to_ram = load_to_ram
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        if load_to_ram:
            print(f"⚠️  Loading entire dataset to RAM")
            self._load_to_ram()
        else:
            print(f"✓ Using lazy loading (memory efficient)")
            self.h5_file = None
            # Get dataset size without loading all data
            with h5py.File(self.file_path, 'r') as f:
                self.N = len(f['images'])
            print(f"✓ Found {self.N} images in dataset")
    
    def _load_to_ram(self) -> None:
        """Load entire dataset into RAM (requires ~6GB)."""
        print(f"Loading {self.file_path} into RAM...")
        
        with h5py.File(self.file_path, 'r') as f:
            self.images = f['images'][:]  # Shape: (N, 256, 256, 3)
            self.labels = f['ans'][:]
        
        # Ensure correct type for PIL (0-255 uint8)
        self.images = self.images.astype(np.uint8)
        self.N = len(self.images)
        
        print(f"✓ Loaded {self.N} images into RAM (~{self.images.nbytes / 1e9:.2f}GB)")
    
    def __len__(self) -> int:
        return self.N
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        """
        Get a single item.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, label)
        """
        if self.load_to_ram:
            # Fast path: data already in RAM
            img = self.images[idx]
            label = self.labels[idx]
        else:
            # Lazy loading: read from disk on demand
            if self.h5_file is None:
                # Open file once and keep it open for efficiency
                self.h5_file = h5py.File(self.file_path, 'r')
            
            img = self.h5_file['images'][idx]
            label = self.h5_file['ans'][idx]
            img = img.astype(np.uint8)
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        
        return img, int(label)
    
    def __del__(self):
        """Close h5 file if open."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


class TransformDataset(Dataset):
    """
    Wraps a dataset or subset and applies a new transform.
    Useful for applying different transforms to train/val splits.
    
    Args:
        subset: Base dataset or subset
        transform: Transform to apply
    
    Example:
        train_ds = TransformDataset(train_subset, train_transform)
        val_ds = TransformDataset(val_subset, val_transform)
    """
    
    def __init__(self, subset: Dataset, transform: Callable):
        self.subset = subset
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.subset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get transformed item."""
        item = self.subset[idx]
        
        if isinstance(item, tuple):
            img, label = item
            return self.transform(img), label
        
        return self.transform(item)

