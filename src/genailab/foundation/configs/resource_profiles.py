"""
Resource profiles for different hardware environments.

Helps automatically select appropriate model configurations and training settings
based on available compute resources.
"""

from dataclasses import dataclass
from typing import Optional
import platform


@dataclass
class ResourceProfile:
    """Hardware resource profile for training configuration."""
    
    # Required fields (no defaults)
    name: str
    device: str  # "mps", "cuda", "cpu"
    memory_gb: float
    recommended_model_size: str  # "small", "medium", "large"
    max_batch_size: int
    use_mixed_precision: bool
    use_gradient_checkpointing: bool
    
    # Optional fields (with defaults)
    num_gpus: int = 1
    use_distributed: bool = False
    use_torch_compile: bool = False
    use_flash_attention: bool = False
    
    def __str__(self) -> str:
        return (
            f"ResourceProfile(name='{self.name}', "
            f"device='{self.device}', "
            f"memory={self.memory_gb}GB, "
            f"recommended_size='{self.recommended_model_size}')"
        )


# ============================================================================
# Preset Profiles
# ============================================================================

M1_PROFILE = ResourceProfile(
    name="M1 MacBook Pro 16GB",
    device="mps",
    memory_gb=16.0,
    num_gpus=0,  # Unified memory, not discrete GPU
    
    # Recommendations
    recommended_model_size="small",
    max_batch_size=8,
    use_mixed_precision=True,
    use_gradient_checkpointing=True,
    use_distributed=False,
    
    # M1-specific
    use_torch_compile=False,  # Limited support on MPS
    use_flash_attention=False,  # Not available on MPS
)

RUNPOD_PROFILE = ResourceProfile(
    name="RunPod RTX 3090 24GB",
    device="cuda",
    memory_gb=24.0,
    num_gpus=1,
    
    # Recommendations
    recommended_model_size="medium",
    max_batch_size=32,
    use_mixed_precision=True,
    use_gradient_checkpointing=False,
    use_distributed=False,
    
    # GPU optimizations
    use_torch_compile=True,
    use_flash_attention=True,
)

CLOUD_PROFILE = ResourceProfile(
    name="Cloud A100 40GB",
    device="cuda",
    memory_gb=40.0,
    num_gpus=1,
    
    # Recommendations
    recommended_model_size="large",
    max_batch_size=64,
    use_mixed_precision=True,
    use_gradient_checkpointing=False,
    use_distributed=False,
    
    # GPU optimizations
    use_torch_compile=True,
    use_flash_attention=True,
)

MULTI_GPU_PROFILE = ResourceProfile(
    name="Multi-GPU Cloud (4x A100 40GB)",
    device="cuda",
    memory_gb=160.0,  # Total across GPUs
    num_gpus=4,
    
    # Recommendations
    recommended_model_size="large",
    max_batch_size=256,  # Total across GPUs
    use_mixed_precision=True,
    use_gradient_checkpointing=False,
    use_distributed=True,
    
    # GPU optimizations
    use_torch_compile=True,
    use_flash_attention=True,
)


def detect_resource_profile() -> ResourceProfile:
    """
    Automatically detect hardware and return appropriate resource profile.
    
    Returns:
        ResourceProfile matching current hardware
        
    Examples:
        >>> profile = detect_resource_profile()
        >>> print(f"Detected: {profile.name}")
        >>> print(f"Recommended model size: {profile.recommended_model_size}")
    """
    import torch
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        num_gpus = torch.cuda.device_count()
        
        # Get GPU memory (first GPU)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Select profile based on memory and GPU count
        if num_gpus > 1:
            return MULTI_GPU_PROFILE
        elif memory_gb >= 35:
            return CLOUD_PROFILE
        elif memory_gb >= 20:
            return RUNPOD_PROFILE
        else:
            # Smaller GPU, use conservative settings
            return ResourceProfile(
                name=f"CUDA GPU {memory_gb:.0f}GB",
                device="cuda",
                memory_gb=memory_gb,
                num_gpus=1,
                recommended_model_size="small",
                max_batch_size=16,
                use_mixed_precision=True,
                use_gradient_checkpointing=True,
                use_torch_compile=True,
                use_flash_attention=True,
            )
    
    elif torch.backends.mps.is_available():
        # M1/M2/M3 Mac
        device = "mps"
        
        # Detect system memory (rough heuristic)
        system = platform.system()
        if system == "Darwin":
            # Try to get actual memory
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                memory_bytes = int(result.stdout.strip())
                memory_gb = memory_bytes / 1e9
            except:
                memory_gb = 16.0  # Default assumption
        else:
            memory_gb = 16.0
        
        return M1_PROFILE
    
    else:
        # CPU fallback
        return ResourceProfile(
            name="CPU Only",
            device="cpu",
            memory_gb=16.0,
            num_gpus=0,
            recommended_model_size="small",
            max_batch_size=4,
            use_mixed_precision=False,
            use_gradient_checkpointing=True,
            use_torch_compile=False,
            use_flash_attention=False,
        )


def get_resource_profile(name: Optional[str] = None) -> ResourceProfile:
    """
    Get a resource profile by name or auto-detect.
    
    Args:
        name: Profile name ("m1", "runpod", "cloud", "multi_gpu") or None for auto-detect
        
    Returns:
        ResourceProfile instance
        
    Examples:
        >>> # Auto-detect
        >>> profile = get_resource_profile()
        
        >>> # Explicit selection
        >>> profile = get_resource_profile("runpod")
    """
    if name is None:
        return detect_resource_profile()
    
    profiles = {
        "m1": M1_PROFILE,
        "runpod": RUNPOD_PROFILE,
        "cloud": CLOUD_PROFILE,
        "multi_gpu": MULTI_GPU_PROFILE,
    }
    
    name_lower = name.lower()
    if name_lower not in profiles:
        raise ValueError(
            f"Unknown profile '{name}'. Choose from: {list(profiles.keys())} "
            f"or use None for auto-detection"
        )
    
    return profiles[name_lower]


def print_resource_info():
    """Print detected resource information and recommendations."""
    import torch
    
    print("\n" + "="*80)
    print("Resource Detection")
    print("="*80)
    
    # Detect profile
    profile = detect_resource_profile()
    
    print(f"\nDetected Profile: {profile.name}")
    print(f"  Device: {profile.device}")
    print(f"  Memory: {profile.memory_gb:.1f} GB")
    print(f"  GPUs: {profile.num_gpus}")
    
    print(f"\nRecommendations:")
    print(f"  Model size: {profile.recommended_model_size}")
    print(f"  Max batch size: {profile.max_batch_size}")
    print(f"  Mixed precision: {profile.use_mixed_precision}")
    print(f"  Gradient checkpointing: {profile.use_gradient_checkpointing}")
    print(f"  Torch compile: {profile.use_torch_compile}")
    print(f"  Flash attention: {profile.use_flash_attention}")
    
    # Additional device info
    if profile.device == "cuda":
        print(f"\nCUDA Info:")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device name: {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
    elif profile.device == "mps":
        print(f"\nMPS Info:")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
        print(f"  MPS built: {torch.backends.mps.is_built()}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    print_resource_info()
    
    # Show all profiles
    print("\nAvailable Profiles:")
    print("-"*80)
    for name, profile in [
        ("M1", M1_PROFILE),
        ("RunPod", RUNPOD_PROFILE),
        ("Cloud", CLOUD_PROFILE),
        ("Multi-GPU", MULTI_GPU_PROFILE),
    ]:
        print(f"\n{name}:")
        print(f"  {profile}")
