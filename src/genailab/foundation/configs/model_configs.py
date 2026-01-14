"""
Model size configurations for resource-aware training.

Provides small/medium/large presets optimized for different hardware:
- SMALL: M1 MacBook Pro 16GB (your current system)
- MEDIUM: RunPod with 24GB GPU
- LARGE: Cloud instances with 40GB+ GPU
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for model architecture and training."""
    
    # Model architecture
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float = 4.0
    
    # Token configuration
    num_tokens: int = 64
    token_dim: Optional[int] = None  # If None, uses embed_dim
    
    # Gene expression specific
    num_genes: int = 20000
    
    # Training
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # Memory optimization
    use_checkpoint: bool = False  # Gradient checkpointing
    use_flash_attention: bool = False
    
    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def hidden_dim(self) -> int:
        """Hidden dimension in feedforward layers."""
        return int(self.embed_dim * self.mlp_ratio)
    
    @property
    def total_params_millions(self) -> float:
        """Approximate total parameters in millions."""
        # Rough estimate for transformer
        # Attention: 4 * embed_dim^2 per head (Q, K, V, O)
        # FFN: 2 * embed_dim * hidden_dim
        # Per layer: attention + FFN + norms
        attn_params = 4 * self.embed_dim * self.embed_dim
        ffn_params = 2 * self.embed_dim * self.hidden_dim
        layer_params = attn_params + ffn_params + 2 * self.embed_dim  # LayerNorms
        
        # Encoder/decoder (rough estimate)
        encoder_params = self.num_genes * self.num_tokens * self.token_dim
        decoder_params = self.num_tokens * self.token_dim * self.num_genes
        
        total = (layer_params * self.depth + encoder_params + decoder_params) / 1e6
        return round(total, 2)
    
    def memory_estimate_gb(self, dtype_bytes: int = 4) -> float:
        """
        Estimate memory usage in GB.
        
        Args:
            dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16)
        """
        # Model parameters
        model_memory = self.total_params_millions * 1e6 * dtype_bytes / 1e9
        
        # Optimizer states (Adam: 2x parameters for momentum + variance)
        optimizer_memory = model_memory * 2
        
        # Activations (rough estimate based on batch size and depth)
        activation_memory = (
            self.batch_size * self.num_tokens * self.embed_dim * 
            self.depth * dtype_bytes / 1e9
        )
        
        # Gradients
        gradient_memory = model_memory
        
        total = model_memory + optimizer_memory + activation_memory + gradient_memory
        
        # Add 20% overhead for PyTorch/CUDA
        total *= 1.2
        
        return round(total, 2)


# ============================================================================
# Preset Configurations
# ============================================================================

SMALL_CONFIG = ModelConfig(
    # Architecture (optimized for M1 MacBook Pro 16GB)
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    
    # Tokens
    num_tokens=64,
    token_dim=256,
    
    # Training
    batch_size=8,  # Small batch for limited memory
    gradient_accumulation_steps=4,  # Effective batch = 32
    mixed_precision=True,
    use_checkpoint=True,  # Enable gradient checkpointing
    use_flash_attention=False,  # Not available on M1
    
    # Gene expression
    num_genes=20000,
)

MEDIUM_CONFIG = ModelConfig(
    # Architecture (optimized for RunPod 24GB GPU)
    embed_dim=512,
    depth=12,
    num_heads=8,
    mlp_ratio=4.0,
    
    # Tokens
    num_tokens=64,
    token_dim=512,
    
    # Training
    batch_size=32,
    gradient_accumulation_steps=1,
    mixed_precision=True,
    use_checkpoint=False,
    use_flash_attention=True,  # Available on modern GPUs
    
    # Gene expression
    num_genes=20000,
)

LARGE_CONFIG = ModelConfig(
    # Architecture (optimized for Cloud 40GB+ GPU)
    embed_dim=768,
    depth=24,
    num_heads=12,
    mlp_ratio=4.0,
    
    # Tokens
    num_tokens=128,  # More tokens for richer representation
    token_dim=768,
    
    # Training
    batch_size=64,
    gradient_accumulation_steps=1,
    mixed_precision=True,
    use_checkpoint=False,
    use_flash_attention=True,
    
    # Gene expression
    num_genes=20000,
)


def get_model_config(size: str = "small") -> ModelConfig:
    """
    Get a preset model configuration.
    
    Args:
        size: One of "small", "medium", "large"
        
    Returns:
        ModelConfig instance
        
    Examples:
        >>> config = get_model_config("small")
        >>> print(f"Model has ~{config.total_params_millions}M parameters")
        >>> print(f"Estimated memory: {config.memory_estimate_gb()}GB")
    """
    configs = {
        "small": SMALL_CONFIG,
        "medium": MEDIUM_CONFIG,
        "large": LARGE_CONFIG,
    }
    
    if size not in configs:
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {list(configs.keys())}"
        )
    
    return configs[size]


# ============================================================================
# Utility Functions
# ============================================================================

def print_config_comparison():
    """Print a comparison table of all configurations."""
    configs = [
        ("Small (M1 16GB)", SMALL_CONFIG),
        ("Medium (RunPod 24GB)", MEDIUM_CONFIG),
        ("Large (Cloud 40GB+)", LARGE_CONFIG),
    ]
    
    print("\n" + "="*80)
    print("Model Configuration Comparison")
    print("="*80)
    
    print(f"\n{'Config':<25} {'Params (M)':<12} {'Memory (GB)':<12} {'Batch':<8} {'Depth':<8}")
    print("-"*80)
    
    for name, config in configs:
        params = config.total_params_millions
        memory = config.memory_estimate_gb(dtype_bytes=2)  # fp16
        batch = config.effective_batch_size
        depth = config.depth
        
        print(f"{name:<25} {params:<12.1f} {memory:<12.1f} {batch:<8} {depth:<8}")
    
    print("="*80)
    print("\nNotes:")
    print("- Memory estimates assume fp16 mixed precision training")
    print("- Small config uses gradient checkpointing to reduce memory")
    print("- Effective batch size includes gradient accumulation")
    print()


if __name__ == "__main__":
    print_config_comparison()
    
    # Example usage
    print("\nExample: Small config for M1 MacBook")
    config = get_model_config("small")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Depth: {config.depth}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Total params: ~{config.total_params_millions}M")
    print(f"  Memory estimate: ~{config.memory_estimate_gb(dtype_bytes=2)}GB (fp16)")
    print(f"  Effective batch size: {config.effective_batch_size}")
