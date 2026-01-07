"""Quick test script for the diffusion module."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from genailab.diffusion import (
    VPSDE,
    SimpleScoreNetwork,
    train_score_network,
    sample_reverse_sde,
)


def generate_swiss_roll(n_samples=1000):
    """Generate 2D Swiss roll dataset."""
    theta = np.sqrt(np.random.rand(n_samples)) * 3 * np.pi
    x = theta * np.cos(theta)
    y = theta * np.sin(theta)
    data = np.stack([x, y], axis=1) / 10.0
    return data


def main():
    # Set seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Generate data
    x0 = generate_swiss_roll(n_samples=2000)
    print(f"Data shape: {x0.shape}")
    
    # Create SDE and model
    sde = VPSDE(beta_min=0.1, beta_max=20.0, T=1.0)
    model = SimpleScoreNetwork(data_dim=2, hidden_dim=256).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\nTraining...")
    losses = train_score_network(
        model=model,
        data=x0,
        sde=sde,
        num_epochs=5000,
        batch_size=128,
        lr=1e-3,
        device=device,
    )
    print(f"Final loss: {losses[-1]:.6f}")
    
    # Sample
    print("\nSampling...")
    samples, trajectory = sample_reverse_sde(
        model=model,
        sde=sde,
        n_samples=2000,
        num_steps=500,
        data_dim=2,
        device=device,
    )
    print(f"Generated {samples.shape[0]} samples")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(losses)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    
    axes[1].scatter(x0[:, 0], x0[:, 1], alpha=0.3, s=1)
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_title('Real Data')
    axes[1].set_aspect('equal')
    
    axes[2].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1, color='orange')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_title('Generated Samples')
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('diffusion_test_output.png', dpi=150)
    plt.show()
    print("Saved plot to diffusion_test_output.png")


if __name__ == "__main__":
    main()
