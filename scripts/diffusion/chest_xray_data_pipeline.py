#!/usr/bin/env python3
"""Chest X-ray Data Pipeline for Diffusion Model Training.

This script handles downloading, preprocessing, and analyzing chest X-ray
datasets from Kaggle for use with diffusion models.

Supported Datasets:
    1. Chest X-Ray Images (Pneumonia) - ~1.3GB
       https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
       
    2. NIH Chest X-rays - ~45GB (larger, more diverse)
       https://www.kaggle.com/datasets/nih-chest-xrays/data

Usage:
    # Download and setup (requires Kaggle API credentials)
    python chest_xray_data_pipeline.py download --dataset pneumonia
    
    # Analyze dataset
    python chest_xray_data_pipeline.py analyze --data-dir data/chest_xray
    
    # Visualize samples
    python chest_xray_data_pipeline.py visualize --data-dir data/chest_xray --n-samples 16
    
    # Preprocess for training
    python chest_xray_data_pipeline.py preprocess --data-dir data/chest_xray --output-dir data/chest_xray_processed

Prerequisites:
    1. Install Kaggle API: pip install kaggle
    2. Setup credentials: 
       - Go to kaggle.com -> Account -> Create New API Token
       - Save kaggle.json to ~/.kaggle/kaggle.json
       - chmod 600 ~/.kaggle/kaggle.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

# Optional imports
try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Constants
# =============================================================================

KAGGLE_DATASETS = {
    "pneumonia": {
        "name": "paultimothymooney/chest-xray-pneumonia",
        "description": "Chest X-Ray Images (Pneumonia) - 5,863 images, ~1.3GB",
        "structure": {
            "train": ["NORMAL", "PNEUMONIA"],
            "val": ["NORMAL", "PNEUMONIA"],
            "test": ["NORMAL", "PNEUMONIA"],
        },
    },
    "nih": {
        "name": "nih-chest-xrays/data",
        "description": "NIH Chest X-rays - 112,120 images, ~45GB",
        "structure": "flat",  # All images in one directory
    },
}

DEFAULT_DATA_DIR = Path("data/chest_xray")


# =============================================================================
# Download Functions
# =============================================================================

def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle credentials not found!")
        print("\nTo setup Kaggle API:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' section")
        print("  3. Click 'Create New Token'")
        print("  4. Save the downloaded kaggle.json to ~/.kaggle/")
        print("  5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    mode = oct(kaggle_json.stat().st_mode)[-3:]
    if mode != "600":
        print(f"⚠️  Kaggle credentials have incorrect permissions: {mode}")
        print("   Run: chmod 600 ~/.kaggle/kaggle.json")
    
    return True


def download_dataset(
    dataset_key: str = "pneumonia",
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download a chest X-ray dataset from Kaggle.
    
    Args:
        dataset_key: Key from KAGGLE_DATASETS ('pneumonia' or 'nih')
        output_dir: Where to save the dataset
        force: Re-download even if exists
        
    Returns:
        Path to the downloaded dataset
    """
    if dataset_key not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Choose from: {list(KAGGLE_DATASETS.keys())}")
    
    dataset_info = KAGGLE_DATASETS[dataset_key]
    dataset_name = dataset_info["name"]
    
    if output_dir is None:
        output_dir = DEFAULT_DATA_DIR
    output_dir = Path(output_dir)
    
    # Check if already downloaded
    if output_dir.exists() and not force:
        n_files = sum(1 for _ in output_dir.rglob("*") if _.is_file())
        if n_files > 0:
            print(f"✓ Dataset already exists at {output_dir} ({n_files} files)")
            print("  Use --force to re-download")
            return output_dir
    
    # Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)
    
    print(f"\n📥 Downloading: {dataset_info['description']}")
    print(f"   From: kaggle.com/datasets/{dataset_name}")
    print(f"   To: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download using Kaggle CLI
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", str(output_dir),
            "--unzip"
        ]
        print(f"\n   Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"\n✓ Download complete: {output_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Download failed: {e}")
        print("   Make sure kaggle is installed: pip install kaggle")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Kaggle CLI not found!")
        print("   Install with: pip install kaggle")
        sys.exit(1)
    
    return output_dir


# =============================================================================
# Analysis Functions
# =============================================================================

def find_images(data_dir: Path) -> List[Path]:
    """Find all image files in a directory."""
    extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    images = []
    for ext in extensions:
        images.extend(data_dir.rglob(f"*{ext}"))
    return sorted(images)


def analyze_dataset(data_dir: Path) -> Dict:
    """Analyze a chest X-ray dataset.
    
    Returns statistics about image sizes, class distribution, etc.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    print(f"\n📊 Analyzing dataset: {data_dir}")
    
    # Find all images
    image_paths = find_images(data_dir)
    print(f"   Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        print("   ❌ No images found!")
        return {}
    
    # Analyze structure
    splits = Counter()
    classes = Counter()
    sizes = []
    file_sizes = []
    
    print("   Scanning images...")
    for img_path in tqdm(image_paths, desc="   Analyzing"):
        # Get split and class from path
        parts = img_path.relative_to(data_dir).parts
        if len(parts) >= 2:
            splits[parts[0]] += 1
            if len(parts) >= 2:
                classes[parts[1]] += 1
        
        # Get image size
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
            file_sizes.append(img_path.stat().st_size)
        except Exception as e:
            print(f"   ⚠️  Could not read {img_path}: {e}")
    
    # Compute statistics
    sizes_arr = np.array(sizes)
    file_sizes_arr = np.array(file_sizes)
    
    stats = {
        "total_images": len(image_paths),
        "total_size_gb": file_sizes_arr.sum() / (1024**3),
        "splits": dict(splits),
        "classes": dict(classes),
        "image_sizes": {
            "min": tuple(sizes_arr.min(axis=0).tolist()),
            "max": tuple(sizes_arr.max(axis=0).tolist()),
            "mean": tuple(sizes_arr.mean(axis=0).astype(int).tolist()),
            "median": tuple(np.median(sizes_arr, axis=0).astype(int).tolist()),
        },
        "file_sizes": {
            "min_kb": file_sizes_arr.min() / 1024,
            "max_kb": file_sizes_arr.max() / 1024,
            "mean_kb": file_sizes_arr.mean() / 1024,
        },
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total images: {stats['total_images']:,}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    
    print("\nSplits:")
    for split, count in sorted(stats['splits'].items()):
        print(f"  {split}: {count:,}")
    
    print("\nClasses:")
    for cls, count in sorted(stats['classes'].items()):
        print(f"  {cls}: {count:,}")
    
    print("\nImage dimensions:")
    print(f"  Min: {stats['image_sizes']['min']}")
    print(f"  Max: {stats['image_sizes']['max']}")
    print(f"  Mean: {stats['image_sizes']['mean']}")
    print(f"  Median: {stats['image_sizes']['median']}")
    
    print("\nFile sizes:")
    print(f"  Min: {stats['file_sizes']['min_kb']:.1f} KB")
    print(f"  Max: {stats['file_sizes']['max_kb']:.1f} KB")
    print(f"  Mean: {stats['file_sizes']['mean_kb']:.1f} KB")
    print("=" * 60)
    
    return stats


def visualize_samples(
    data_dir: Path,
    n_samples: int = 16,
    output_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """Visualize sample images from the dataset."""
    data_dir = Path(data_dir)
    image_paths = find_images(data_dir)
    
    if len(image_paths) == 0:
        print("❌ No images found!")
        return
    
    # Sample images
    n_samples = min(n_samples, len(image_paths))
    indices = np.random.choice(len(image_paths), n_samples, replace=False)
    sample_paths = [image_paths[i] for i in indices]
    
    # Create grid
    rows = int(np.ceil(np.sqrt(n_samples)))
    cols = int(np.ceil(n_samples / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, img_path in enumerate(sample_paths):
        img = Image.open(img_path).convert('L')
        axes[i].imshow(np.array(img), cmap='gray')
        axes[i].axis('off')
        
        # Get label from path
        parts = img_path.relative_to(data_dir).parts
        label = parts[1] if len(parts) >= 2 else "unknown"
        axes[i].set_title(label, fontsize=8)
    
    # Hide unused axes
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Sample Chest X-rays from {data_dir.name}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_size_distribution(data_dir: Path, output_path: Optional[Path] = None) -> None:
    """Plot distribution of image sizes."""
    image_paths = find_images(data_dir)
    
    sizes = []
    for img_path in tqdm(image_paths, desc="Reading sizes"):
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
        except:
            pass
    
    sizes = np.array(sizes)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(sizes[:, 0], bins=50, alpha=0.7, label='Width')
    axes[0].hist(sizes[:, 1], bins=50, alpha=0.7, label='Height')
    axes[0].set_xlabel('Pixels')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Image Dimension Distribution')
    axes[0].legend()
    
    axes[1].scatter(sizes[:, 0], sizes[:, 1], alpha=0.3, s=5)
    axes[1].set_xlabel('Width')
    axes[1].set_ylabel('Height')
    axes[1].set_title('Width vs Height')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved size distribution to {output_path}")
    
    plt.show()


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_dataset(
    data_dir: Path,
    output_dir: Path,
    target_size: int = 128,
    normalize: bool = True,
    split: Optional[str] = None,
) -> None:
    """Preprocess images for training.
    
    Args:
        data_dir: Source directory with raw images
        output_dir: Where to save processed images
        target_size: Target image size (square)
        normalize: Whether to normalize to [0, 1]
        split: Only process specific split (train/val/test)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    print(f"\n🔧 Preprocessing dataset")
    print(f"   Source: {data_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Target size: {target_size}x{target_size}")
    
    # Find images
    if split:
        image_paths = find_images(data_dir / split)
    else:
        image_paths = find_images(data_dir)
    
    print(f"   Found {len(image_paths)} images")
    
    # Process images
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in tqdm(image_paths, desc="   Processing"):
        try:
            # Load and convert to grayscale
            img = Image.open(img_path).convert('L')
            
            # Resize (maintain aspect ratio, then center crop)
            w, h = img.size
            scale = target_size / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            
            # Center crop
            left = (new_w - target_size) // 2
            top = (new_h - target_size) // 2
            img = img.crop((left, top, left + target_size, top + target_size))
            
            # Preserve directory structure
            rel_path = img_path.relative_to(data_dir)
            out_path = output_dir / rel_path.with_suffix('.png')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            img.save(out_path)
            
        except Exception as e:
            print(f"   ⚠️  Failed to process {img_path}: {e}")
    
    print(f"\n✓ Preprocessing complete: {output_dir}")


# =============================================================================
# PyTorch Dataset Integration
# =============================================================================

def test_dataloader(data_dir: Path, batch_size: int = 8, img_size: int = 128) -> None:
    """Test loading data with PyTorch DataLoader."""
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Install with: pip install torch")
        return
    
    from genailab.diffusion import ChestXRayDataset
    
    # Find a subdirectory with images
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if subdirs:
        # Try train/NORMAL first
        test_dir = data_dir / "train" / "NORMAL"
        if not test_dir.exists():
            test_dir = subdirs[0]
    else:
        test_dir = data_dir
    
    print(f"\n🔄 Testing DataLoader with {test_dir}")
    
    try:
        dataset = ChestXRayDataset(root_dir=str(test_dir), img_size=img_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        batch = next(iter(dataloader))
        print(f"   Batch shape: {batch.shape}")
        print(f"   Value range: [{batch.min():.2f}, {batch.max():.2f}]")
        print(f"   ✓ DataLoader working correctly!")
        
        # Visualize batch
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(min(8, batch.shape[0])):
            img = batch[i, 0].numpy()
            img = (img + 1) / 2  # Denormalize
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')
        
        plt.suptitle('Sample Batch from DataLoader')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"   ❌ DataLoader test failed: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chest X-ray Data Pipeline for Diffusion Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download dataset from Kaggle")
    download_parser.add_argument(
        "--dataset", "-d",
        choices=list(KAGGLE_DATASETS.keys()),
        default="pneumonia",
        help="Dataset to download (default: pneumonia)"
    )
    download_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Output directory"
    )
    download_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download"
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze dataset")
    analyze_parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory to analyze"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save stats to JSON file"
    )
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Visualize samples")
    viz_parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory"
    )
    viz_parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=16,
        help="Number of samples to show"
    )
    viz_parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save visualization to file"
    )
    viz_parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plot (just save)"
    )
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess images")
    preprocess_parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Source data directory"
    )
    preprocess_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        required=True,
        help="Output directory for processed images"
    )
    preprocess_parser.add_argument(
        "--size", "-s",
        type=int,
        default=128,
        help="Target image size (default: 128)"
    )
    preprocess_parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        help="Only process specific split"
    )
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test PyTorch DataLoader")
    test_parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Data directory"
    )
    test_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=8,
        help="Batch size"
    )
    test_parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Image size"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n📋 Available datasets:")
        for key, info in KAGGLE_DATASETS.items():
            print(f"   {key}: {info['description']}")
        return
    
    if args.command == "download":
        download_dataset(args.dataset, args.output_dir, args.force)
        
    elif args.command == "analyze":
        stats = analyze_dataset(args.data_dir)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\n✓ Stats saved to {args.output}")
            
    elif args.command == "visualize":
        visualize_samples(
            args.data_dir,
            args.n_samples,
            args.output,
            show=not args.no_show
        )
        
    elif args.command == "preprocess":
        preprocess_dataset(
            args.data_dir,
            args.output_dir,
            args.size,
            split=args.split
        )
        
    elif args.command == "test":
        test_dataloader(args.data_dir, args.batch_size, args.img_size)


if __name__ == "__main__":
    main()
