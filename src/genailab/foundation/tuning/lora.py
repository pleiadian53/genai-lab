"""
LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning.

Reference: Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any
import math


class LoRALinear(nn.Module):
    """
    LoRA-augmented linear layer.
    
    Replaces W with W + ΔW where ΔW = BA (low-rank decomposition).
    Only A and B are trainable, W is frozen.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: Rank of low-rank decomposition (r)
        alpha: Scaling factor (typically 2*rank)
        dropout: Dropout probability for LoRA path
        merge_weights: Whether to merge LoRA weights into base weights
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        merge_weights: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.merge_weights = merge_weights
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # LoRA parameters (trainable)
        # A: (in_features, rank) - initialized with Kaiming uniform
        # B: (rank, out_features) - initialized to zero
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize
        self.reset_parameters()
        
        # Merged state
        self.merged = False
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform (like nn.Linear)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # Initialize B to zero (so ΔW = 0 initially)
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Original output
        result = self.linear(x)
        
        if not self.merged:
            # LoRA path: x @ A @ B
            lora_output = x @ self.lora_A @ self.lora_B
            lora_output = self.dropout(lora_output)
            result = result + lora_output * self.scaling
        
        return result
    
    def merge_weights(self):
        """Merge LoRA weights into base weights for inference."""
        if not self.merged:
            # W_eff = W + α/r * A @ B
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.linear.weight.data += delta_w.T
            self.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights."""
        if self.merged:
            # W = W_eff - α/r * A @ B
            delta_w = (self.lora_A @ self.lora_B) * self.scaling
            self.linear.weight.data -= delta_w.T
            self.merged = False
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"rank={self.rank}, "
            f"alpha={self.alpha}, "
            f"merged={self.merged}"
        )


class LoRA:
    """
    LoRA wrapper for applying low-rank adaptation to models.
    
    Examples:
        >>> # Apply LoRA to specific layers
        >>> model = MyTransformer()
        >>> lora_model = LoRA.apply(
        ...     model,
        ...     target_modules=["attention.query", "attention.key", "attention.value"],
        ...     rank=8,
        ...     alpha=16
        ... )
        
        >>> # Train only LoRA parameters
        >>> optimizer = torch.optim.Adam(lora_model.parameters(), lr=1e-4)
    """
    
    @staticmethod
    def apply(
        model: nn.Module,
        target_modules: List[str],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        verbose: bool = True,
    ) -> nn.Module:
        """
        Apply LoRA to specified modules in a model.
        
        Args:
            model: Model to apply LoRA to
            target_modules: List of module names to replace (supports wildcards)
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: Dropout probability
            verbose: Print information about replaced modules
            
        Returns:
            Model with LoRA applied
        """
        replaced_count = 0
        
        for name, module in model.named_modules():
            # Check if this module matches any target pattern
            if any(target in name for target in target_modules):
                # Only replace nn.Linear layers
                if isinstance(module, nn.Linear):
                    # Get parent module and attribute name
                    parent_name = ".".join(name.split(".")[:-1])
                    attr_name = name.split(".")[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    # Create LoRA layer
                    lora_layer = LoRALinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    )
                    
                    # Copy original weights
                    lora_layer.linear.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_layer.linear.bias.data = module.bias.data.clone()
                    
                    # Replace module
                    setattr(parent, attr_name, lora_layer)
                    replaced_count += 1
                    
                    if verbose:
                        print(f"  Replaced {name} with LoRALinear(rank={rank})")
        
        if verbose:
            print(f"\nApplied LoRA to {replaced_count} layers")
            LoRA.print_trainable_parameters(model)
        
        return model
    
    @staticmethod
    def merge_and_save(model: nn.Module, path: str):
        """
        Merge LoRA weights and save model.
        
        Args:
            model: Model with LoRA layers
            path: Path to save merged model
        """
        # Merge all LoRA layers
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.merge_weights()
        
        # Save
        torch.save(model.state_dict(), path)
        
        # Unmerge (restore original state)
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.unmerge_weights()
    
    @staticmethod
    def save_lora_only(model: nn.Module, path: str):
        """
        Save only LoRA parameters (not base weights).
        
        Args:
            model: Model with LoRA layers
            path: Path to save LoRA parameters
        """
        lora_state_dict = {}
        
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
        
        torch.save(lora_state_dict, path)
    
    @staticmethod
    def load_lora_only(model: nn.Module, path: str):
        """
        Load LoRA parameters into model.
        
        Args:
            model: Model with LoRA layers
            path: Path to LoRA parameters
        """
        lora_state_dict = torch.load(path)
        
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state_dict:
                    module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    
    @staticmethod
    def print_trainable_parameters(model: nn.Module):
        """Print number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        
        for name, param in model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / all_params if all_params > 0 else 0
        
        print(f"\nTrainable parameters:")
        print(f"  Total: {all_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Percentage: {percentage:.2f}%")


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    verbose: bool = True,
) -> nn.Module:
    """
    Convenience function to apply LoRA to a model.
    
    Args:
        model: Model to apply LoRA to
        target_modules: List of module names (default: attention layers)
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
        verbose: Print information
        
    Returns:
        Model with LoRA applied
        
    Examples:
        >>> model = MyTransformer()
        >>> model = apply_lora_to_model(model, rank=8)
    """
    if target_modules is None:
        # Default: apply to attention Q, K, V projections
        target_modules = ["attention.query", "attention.key", "attention.value"]
    
    return LoRA.apply(
        model=model,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Example usage
    print("LoRA Example")
    print("="*80)
    
    # Create a simple model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=256):
            super().__init__()
            self.attention = nn.ModuleDict({
                'query': nn.Linear(d_model, d_model),
                'key': nn.Linear(d_model, d_model),
                'value': nn.Linear(d_model, d_model),
                'output': nn.Linear(d_model, d_model),
            })
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
        
        def forward(self, x):
            return x
    
    # Create model
    model = SimpleTransformer()
    print("\nOriginal model:")
    LoRA.print_trainable_parameters(model)
    
    # Apply LoRA
    print("\n" + "="*80)
    print("Applying LoRA to attention layers...")
    print("="*80)
    model = apply_lora_to_model(
        model,
        target_modules=["attention.query", "attention.key", "attention.value"],
        rank=8,
        alpha=16,
    )
    
    # Test forward pass
    x = torch.randn(2, 10, 256)
    output = model(x)
    print(f"\nForward pass successful: {output.shape}")
