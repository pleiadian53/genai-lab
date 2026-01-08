"""Datasets for diffusion model training.

This module provides both synthetic and real dataset loaders for
training diffusion models on medical imaging data.
"""

from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticXRayDataset(Dataset):
    """Synthetic X-ray-like images for testing the diffusion pipeline.
    
    Generates diverse synthetic chest X-ray images with randomized:
    - Lung positions, sizes, and shapes (elliptical)
    - Heart position and size
    - Rib spacing and intensity
    - Background noise patterns
    - Overall brightness/contrast
    
    Args:
        n_samples: Number of synthetic images to generate
        img_size: Image dimensions (square images)
        seed: Random seed for reproducibility
        
    Example:
        >>> dataset = SyntheticXRayDataset(n_samples=100, img_size=128)
        >>> img = dataset[0]  # Returns tensor of shape (1, 128, 128) in [-1, 1]
    """
    
    def __init__(
        self, 
        n_samples: int = 1000, 
        img_size: int = 128, 
        seed: Optional[int] = 42
    ):
        self.n_samples = n_samples
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)
        self.images = self._generate_synthetic_xrays()
    
    def _generate_synthetic_xrays(self) -> np.ndarray:
        """Generate diverse X-ray-like images with randomized anatomical structures."""
        images = []
        
        for _ in range(self.n_samples):
            img = self._generate_single_xray()
            images.append(img)
        
        return np.array(images, dtype=np.float32)
    
    def _generate_single_xray(self) -> np.ndarray:
        """Generate a single synthetic X-ray with randomized features."""
        size = self.img_size
        
        # Random background intensity and noise level
        bg_intensity = self.rng.uniform(0.4, 0.6)
        noise_level = self.rng.uniform(0.05, 0.15)
        img = self.rng.randn(size, size) * noise_level + bg_intensity
        
        y, x = np.ogrid[:size, :size]
        
        # Randomized lung parameters
        # Left lung
        cx1 = int(size * self.rng.uniform(0.25, 0.40))
        cy1 = int(size * self.rng.uniform(0.40, 0.60))
        rx1 = int(size * self.rng.uniform(0.18, 0.28))  # x radius
        ry1 = int(size * self.rng.uniform(0.22, 0.35))  # y radius (taller)
        lung_darkness1 = self.rng.uniform(0.5, 0.7)
        
        # Elliptical mask for left lung
        mask1 = ((x - cx1)**2 / max(rx1**2, 1) + (y - cy1)**2 / max(ry1**2, 1)) < 1
        img[mask1] *= lung_darkness1
        
        # Right lung
        cx2 = int(size * self.rng.uniform(0.60, 0.75))
        cy2 = int(size * self.rng.uniform(0.40, 0.60))
        rx2 = int(size * self.rng.uniform(0.18, 0.28))
        ry2 = int(size * self.rng.uniform(0.22, 0.35))
        lung_darkness2 = self.rng.uniform(0.5, 0.7)
        
        mask2 = ((x - cx2)**2 / max(rx2**2, 1) + (y - cy2)**2 / max(ry2**2, 1)) < 1
        img[mask2] *= lung_darkness2
        
        # Randomized heart (between lungs, slightly left of center)
        cx3 = int(size * self.rng.uniform(0.42, 0.55))
        cy3 = int(size * self.rng.uniform(0.45, 0.60))
        r3 = int(size * self.rng.uniform(0.08, 0.15))
        heart_brightness = self.rng.uniform(1.2, 1.5)
        
        mask3 = ((x - cx3)**2 + (y - cy3)**2) < r3**2
        img[mask3] *= heart_brightness
        
        # Randomized ribs
        n_ribs = self.rng.randint(4, 8)
        rib_start = int(size * self.rng.uniform(0.15, 0.25))
        rib_spacing = int(size * self.rng.uniform(0.08, 0.12))
        rib_thickness = max(1, int(size * self.rng.uniform(0.01, 0.025)))
        rib_brightness = self.rng.uniform(1.1, 1.3)
        
        for i in range(n_ribs):
            rib_y = rib_start + i * rib_spacing
            if rib_y + rib_thickness < size:
                # Add slight curve to ribs
                curve = np.sin(np.linspace(0, np.pi, size)) * self.rng.uniform(0, 3)
                for dy in range(-rib_thickness // 2, rib_thickness // 2 + 1):
                    for xi in range(size):
                        yi = int(rib_y + dy + curve[xi])
                        if 0 <= yi < size:
                            img[yi, xi] *= rib_brightness
        
        # Optional: Add spine (vertical structure in center)
        if self.rng.random() > 0.3:
            spine_x = int(size * self.rng.uniform(0.47, 0.53))
            spine_width = max(2, int(size * self.rng.uniform(0.03, 0.06)))
            spine_brightness = self.rng.uniform(1.1, 1.25)
            img[:, spine_x - spine_width // 2:spine_x + spine_width // 2] *= spine_brightness
        
        # Optional: Add clavicles (horizontal structures at top)
        if self.rng.random() > 0.4:
            clav_y = int(size * self.rng.uniform(0.12, 0.20))
            clav_brightness = self.rng.uniform(1.15, 1.3)
            clav_thickness = max(1, int(size * 0.015))
            img[clav_y:clav_y + clav_thickness, :] *= clav_brightness
        
        # Random global contrast adjustment
        contrast = self.rng.uniform(0.8, 1.2)
        img = (img - 0.5) * contrast + 0.5
        
        # Clip to valid range
        img = np.clip(img, 0, 1)
        
        return img
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.images[idx]
        img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dim
        img = img * 2 - 1  # Normalize to [-1, 1]
        return img


class ChestXRayDataset(Dataset):
    """Dataset loader for real chest X-ray images.
    
    Supports loading from local directories containing X-ray images.
    Compatible with Kaggle chest X-ray datasets.
    
    Args:
        root_dir: Path to directory containing images
        img_size: Target image size (will resize)
        transform: Optional additional transforms
        
    Example:
        >>> dataset = ChestXRayDataset(
        ...     root_dir="data/chest_xray/train/NORMAL",
        ...     img_size=128
        ... )
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: int = 128,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Find all image files
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            self.image_paths.extend(self.root_dir.glob(ext))
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        from PIL import Image
        
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Convert to numpy
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Apply optional transform
        if self.transform:
            img = self.transform(img)
        
        # Convert to tensor
        img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dim
        img = img * 2 - 1  # Normalize to [-1, 1]
        
        return img


def get_dataset(
    name: str,
    img_size: int = 128,
    n_samples: int = 1000,
    root_dir: Optional[str] = None,
    **kwargs
) -> Dataset:
    """Factory function to get a dataset by name.
    
    Args:
        name: Dataset name ('synthetic', 'chest_xray', etc.)
        img_size: Image size
        n_samples: Number of samples (for synthetic)
        root_dir: Root directory (for real datasets)
        **kwargs: Additional arguments passed to dataset
        
    Returns:
        Dataset instance
    """
    if name == 'synthetic':
        return SyntheticXRayDataset(
            n_samples=n_samples,
            img_size=img_size,
            **kwargs
        )
    elif name == 'chest_xray':
        if root_dir is None:
            raise ValueError("root_dir required for chest_xray dataset")
        return ChestXRayDataset(
            root_dir=root_dir,
            img_size=img_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
