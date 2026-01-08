#!/usr/bin/env python3
"""
Medical Imaging Diffusion Model - Driver Script

This script trains a diffusion model on medical images (synthetic or real).
It mirrors the notebook: notebooks/diffusion/03_medical_imaging_diffusion/

Usage:
    # Quick test on M1 Mac (tiny model, few epochs)
    python scripts/diffusion/03_medical_imaging_diffusion.py --preset tiny --epochs 10

    # Small model for local testing
    python scripts/diffusion/03_medical_imaging_diffusion.py --preset small --epochs 100

    # Full training on GPU (A40, etc.)
    python scripts/diffusion/03_medical_imaging_diffusion.py --preset large --epochs 5000

    # Custom configuration
    python scripts/diffusion/03_medical_imaging_diffusion.py \
        --base-channels 32 \
        --channel-mults 1,2,4 \
        --num-res-blocks 1 \
        --epochs 500 \
        --batch-size 16

Model Size Presets:
    tiny:   ~1M params   - Quick logic verification (M1 Mac friendly)
    small:  ~5M params   - Local testing with reasonable quality
    medium: ~20M params  - Good quality, needs decent GPU
    large:  ~50M params  - Production quality (A40/A100 recommended)
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Import from genailab
from genailab.diffusion import VPSDE, UNet2D, train_image_diffusion, SyntheticXRayDataset
from genailab import get_config, get_checkpoint_dir, get_device


# =============================================================================
# Model Size Presets
# =============================================================================

@dataclass
class ModelPreset:
    """Configuration preset for different model sizes."""
    name: str
    base_channels: int
    channel_multipliers: Tuple[int, ...]
    num_res_blocks: int
    time_emb_dim: int
    img_size: int
    batch_size: int
    default_epochs: int
    description: str


PRESETS = {
    "tiny": ModelPreset(
        name="tiny",
        base_channels=16,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_emb_dim=64,
        img_size=64,
        batch_size=32,
        default_epochs=100,
        description="~1M params - Quick logic verification (M1 Mac friendly)",
    ),
    "small": ModelPreset(
        name="small",
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        num_res_blocks=1,
        time_emb_dim=128,
        img_size=64,
        batch_size=32,
        default_epochs=500,
        description="~5M params - Local testing with reasonable quality",
    ),
    "medium": ModelPreset(
        name="medium",
        base_channels=48,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=192,
        img_size=128,
        batch_size=16,
        default_epochs=2000,
        description="~20M params - Good quality, needs decent GPU",
    ),
    "large": ModelPreset(
        name="large",
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        num_res_blocks=2,
        time_emb_dim=256,
        img_size=128,
        batch_size=32,
        default_epochs=5000,
        description="~50M params - Production quality (A40/A100 recommended)",
    ),
}


# =============================================================================
# Sampling
# =============================================================================

@torch.no_grad()
def sample_images(
    model: nn.Module,
    sde: VPSDE,
    n_samples: int = 16,
    img_size: int = 128,
    num_steps: int = 500,
    device: str = 'cuda',
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample images using reverse SDE."""
    model.eval()
    
    x = torch.randn(n_samples, 1, img_size, img_size, device=device)
    dt = -sde.T / num_steps
    trajectory = [x.cpu().numpy()]
    
    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        t = sde.T - i * (-dt)
        t_batch = torch.ones(n_samples, device=device) * t
        
        score = model(x, t_batch)
        
        drift = sde.drift(x.cpu().numpy(), t)
        drift = torch.FloatTensor(drift).to(device)
        g_t = sde.diffusion(t)
        drift = drift - (g_t ** 2) * score
        
        noise = torch.randn_like(x)
        diffusion = g_t * noise * np.sqrt(-dt)
        
        x = x + drift * dt + diffusion
        
        if i % 50 == 0:
            trajectory.append(x.cpu().numpy())
    
    return x.cpu().numpy(), np.array(trajectory)


# =============================================================================
# Main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train diffusion model on medical images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Preset selection
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=list(PRESETS.keys()),
        default="small",
        help="Model size preset (default: small)",
    )
    
    # Override preset values
    parser.add_argument("--base-channels", type=int, help="Base channel count")
    parser.add_argument("--channel-mults", type=str, help="Channel multipliers, comma-separated (e.g., 1,2,4,8)")
    parser.add_argument("--num-res-blocks", type=int, help="Residual blocks per resolution")
    parser.add_argument("--time-emb-dim", type=int, help="Time embedding dimension")
    parser.add_argument("--img-size", type=int, help="Image size (square)")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    
    # Training
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of training samples")
    parser.add_argument("--save-every", type=int, default=500, help="Save checkpoint every N epochs")
    
    # Sampling
    parser.add_argument("--sample", action="store_true", help="Generate samples after training")
    parser.add_argument("--sample-steps", type=int, default=500, help="Number of sampling steps")
    parser.add_argument("--n-gen", type=int, default=16, help="Number of images to generate")
    
    # Output
    parser.add_argument("--output-dir", type=str, help="Output directory (default: checkpoints/diffusion/medical_imaging)")
    parser.add_argument("--experiment-name", type=str, default="medical_diffusion", help="Experiment name for logging")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, help="Device (auto-detected if not specified)")
    parser.add_argument("--list-presets", action="store_true", help="List available presets and exit")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List presets and exit
    if args.list_presets:
        print("\nAvailable Model Presets:")
        print("=" * 60)
        for name, preset in PRESETS.items():
            print(f"\n  {name}:")
            print(f"    {preset.description}")
            print(f"    base_channels: {preset.base_channels}")
            print(f"    channel_multipliers: {preset.channel_multipliers}")
            print(f"    num_res_blocks: {preset.num_res_blocks}")
            print(f"    img_size: {preset.img_size}")
            print(f"    default_epochs: {preset.default_epochs}")
        print()
        return
    
    # Get preset
    preset = PRESETS[args.preset]
    
    # Override with command-line args
    base_channels = args.base_channels or preset.base_channels
    channel_mults = tuple(map(int, args.channel_mults.split(","))) if args.channel_mults else preset.channel_multipliers
    num_res_blocks = args.num_res_blocks or preset.num_res_blocks
    time_emb_dim = args.time_emb_dim or preset.time_emb_dim
    img_size = args.img_size or preset.img_size
    batch_size = args.batch_size or preset.batch_size
    epochs = args.epochs or preset.default_epochs
    
    # Device
    device = args.device or get_device()
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_checkpoint_dir(f"diffusion/medical_imaging/{args.experiment_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Medical Imaging Diffusion Model Training")
    print("=" * 60)
    print(f"\nPreset: {args.preset} ({preset.description})")
    print(f"\nModel Configuration:")
    print(f"  base_channels: {base_channels}")
    print(f"  channel_multipliers: {channel_mults}")
    print(f"  num_res_blocks: {num_res_blocks}")
    print(f"  time_emb_dim: {time_emb_dim}")
    print(f"  img_size: {img_size}x{img_size}")
    print(f"\nTraining Configuration:")
    print(f"  epochs: {epochs}")
    print(f"  batch_size: {batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  n_samples: {args.n_samples}")
    print(f"\nDevice: {device}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")
    
    # Save config
    config = {
        "preset": args.preset,
        "base_channels": base_channels,
        "channel_multipliers": list(channel_mults),
        "num_res_blocks": num_res_blocks,
        "time_emb_dim": time_emb_dim,
        "img_size": img_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": args.lr,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "device": str(device),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Create dataset
    print("Creating dataset...")
    dataset = SyntheticXRayDataset(n_samples=args.n_samples, img_size=img_size, seed=args.seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"  Dataset: {len(dataset)} images, shape: {dataset[0].shape}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # Create model
    print("\nCreating model...")
    model = UNet2D(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        channel_multipliers=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Test forward pass
    print("  Testing forward pass...")
    x_test = torch.randn(2, 1, img_size, img_size).to(device)
    t_test = torch.rand(2).to(device)
    with torch.no_grad():
        out_test = model(x_test, t_test)
    print(f"  Input: {x_test.shape} -> Output: {out_test.shape}")
    print("  ✓ Model initialized successfully")
    
    # Create SDE
    sde = VPSDE(schedule='cosine', T=1.0)
    print(f"\nSDE: VP-SDE with {sde.schedule_name} schedule")
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Estimated time: ~{epochs * len(dataloader) / 60:.0f} minutes on {device}")
    
    losses = train_image_diffusion(
        model=model,
        dataloader=dataloader,
        sde=sde,
        num_epochs=epochs,
        lr=args.lr,
        device=device,
        save_every=args.save_every,
        checkpoint_dir=output_dir,
    )
    
    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_loss': losses[-1] if losses else None,
    }, final_path)
    print(f"\n✓ Final model saved to: {final_path}")
    
    # Save losses
    np.save(output_dir / "losses.npy", np.array(losses))
    print(f"✓ Training losses saved to: {output_dir / 'losses.npy'}")
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Best loss: {min(losses):.6f} at epoch {np.argmin(losses) + 1}")
    
    # Save training curve plot
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curve saved to: {output_dir / 'training_curve.png'}")
    
    # Generate samples
    if args.sample:
        print(f"\nGenerating {args.n_gen} samples...")
        samples, trajectory = sample_images(
            model=model,
            sde=sde,
            n_samples=args.n_gen,
            img_size=img_size,
            num_steps=args.sample_steps,
            device=device,
        )
        
        np.save(output_dir / "samples.npy", samples)
        print(f"✓ Samples saved to: {output_dir / 'samples.npy'}")
        
        # Save generated images grid
        n_display = min(16, args.n_gen)
        rows = int(np.ceil(np.sqrt(n_display)))
        cols = int(np.ceil(n_display / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = np.atleast_2d(axes).flatten()
        
        for i in range(n_display):
            img = samples[i, 0]
            img = (img + 1) / 2  # Denormalize
            img = np.clip(img, 0, 1)
            axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
            axes[i].axis('off')
        
        for i in range(n_display, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Generated Samples', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "generated_samples.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Generated samples grid saved to: {output_dir / 'generated_samples.png'}")
    
    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
