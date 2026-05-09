#!/usr/bin/env python
"""End-to-end test script for mini_vit_from_scratch.ipynb.

Tests all architecture components and one training/eval cycle.
Does NOT run 20 epochs — uses 3 mini-batches to verify the loop.
"""
import sys
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Optional

errors_log = []


def log_step(name: str) -> None:
    print(f"\n{'='*60}\n  STEP: {name}\n{'='*60}")


def log_ok(msg: str = "OK") -> None:
    print(f"  ✓ {msg}")


def log_error(step: str, exc: Exception) -> None:
    tb = traceback.format_exc()
    errors_log.append({"step": step, "error": str(exc), "traceback": tb})
    print(f"  ✗ ERROR: {exc}\n{tb}")


# ── Paste model definitions (copied from notebook) ─────────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_scores.softmax(dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out, (attn_weights if return_attention else None)


class MLP(nn.Module):
    def __init__(self, embed_dim=128, mlp_dim=256, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_dim=256, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout=dropout, proj_dropout=dropout)
        self.mlp = MLP(embed_dim, mlp_dim, dropout)

    def forward(self, x, return_attention=False):
        attn_out, attn_weights = self.attn(self.norm1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class MiniViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=4, num_heads=4, mlp_dim=256,
                 dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, return_attention=False):
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.emb_dropout(tokens)
        all_attn = []
        for block in self.blocks:
            tokens, attn_weights = block(tokens, return_attention=return_attention)
            if return_attention:
                all_attn.append(attn_weights)
        cls_output = tokens[:, 0]
        cls_output = self.norm(cls_output)
        logits = self.head(cls_output)
        return logits, (all_attn if return_attention else None)


# ── Tests ───────────────────────────────────────────────────────────────────

log_step("Device setup")
try:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log_ok(f"device = {device}")
    log_ok(f"PyTorch {torch.__version__}")
except Exception as e:
    log_error("Device setup", e)


log_step("PatchEmbedding — shape check")
try:
    pe = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=128)
    dummy = torch.randn(2, 3, 32, 32)
    out = pe(dummy)
    assert out.shape == (2, 64, 128), f"Expected (2,64,128), got {out.shape}"
    log_ok(f"(2,3,32,32) → {out.shape}  ✓")
except Exception as e:
    log_error("PatchEmbedding", e)


log_step("MultiHeadSelfAttention — shapes and attention row sums")
try:
    mhsa = MultiHeadSelfAttention(embed_dim=128, num_heads=4)
    dummy_seq = torch.randn(2, 65, 128)
    out, attn = mhsa(dummy_seq, return_attention=True)
    assert out.shape == (2, 65, 128), f"Expected (2,65,128), got {out.shape}"
    assert attn.shape == (2, 4, 65, 65), f"Expected (2,4,65,65), got {attn.shape}"
    row_sum = attn[0, 0, 0].sum().item()
    assert abs(row_sum - 1.0) < 1e-4, f"Attention row sum {row_sum} ≠ 1.0"
    log_ok(f"out {out.shape}, attn {attn.shape}, row_sum={row_sum:.4f}  ✓")
except Exception as e:
    log_error("MultiHeadSelfAttention", e)


log_step("MLP — shape check")
try:
    mlp = MLP(embed_dim=128, mlp_dim=256)
    dummy_seq = torch.randn(2, 65, 128)
    out = mlp(dummy_seq)
    assert out.shape == (2, 65, 128), f"Expected (2,65,128), got {out.shape}"
    log_ok(f"(2,65,128) → {out.shape}  ✓")
except Exception as e:
    log_error("MLP", e)


log_step("TransformerBlock — shape check")
try:
    block = TransformerBlock(embed_dim=128, num_heads=4, mlp_dim=256)
    dummy_seq = torch.randn(2, 65, 128)
    out, attn = block(dummy_seq, return_attention=True)
    assert out.shape == (2, 65, 128), f"Expected (2,65,128), got {out.shape}"
    log_ok(f"out {out.shape}, attn {attn.shape}  ✓")
except Exception as e:
    log_error("TransformerBlock", e)


log_step("MiniViT — parameter count and forward pass")
try:
    model = MiniViT(image_size=32, patch_size=4, in_channels=3, num_classes=10,
                    embed_dim=128, depth=4, num_heads=4, mlp_dim=256,
                    dropout=0.1, emb_dropout=0.1)
    total = sum(p.numel() for p in model.parameters())
    log_ok(f"Parameters: {total:,}  (~{total*4/1024**2:.1f} MB FP32)")
    dummy_images = torch.randn(4, 3, 32, 32)
    logits, attn_list = model(dummy_images, return_attention=True)
    assert logits.shape == (4, 10), f"Expected (4,10), got {logits.shape}"
    assert len(attn_list) == 4, f"Expected 4 attention maps, got {len(attn_list)}"
    log_ok(f"logits {logits.shape}, {len(attn_list)} attention maps  ✓")
except Exception as e:
    log_error("MiniViT forward pass", e)


log_step("MiniViT on device — forward pass")
try:
    model = model.to(device)
    dummy_images = torch.randn(4, 3, 32, 32).to(device)
    logits, _ = model(dummy_images)
    assert logits.shape == (4, 10)
    log_ok(f"Forward pass on {device}  ✓")
except Exception as e:
    log_error("MiniViT on device", e)


log_step("Data loading — CIFAR-10 (real data) or synthetic fallback")
train_loader = test_loader = train_dataset = test_dataset = None
try:
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
    # pin_memory should be False for MPS (CUDA-only feature)
    pin = device.type == "cuda"
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    # num_workers=0 required in a plain .py script on macOS (spawn start method).
    # num_workers>0 works fine in Jupyter because the notebook is already in a
    # proper main process context.
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=pin)
    log_ok(f"CIFAR-10 loaded: train={len(train_dataset):,}  test={len(test_dataset):,}")
    log_ok(f"pin_memory={pin}  (correct for {device.type})")
except Exception as e:
    # Network/server unavailable — fall back to synthetic data for testing
    print(f"  ⚠ CIFAR-10 unavailable ({e}). Using synthetic data instead.")
    from torch.utils.data import TensorDataset
    _x = torch.randn(256, 3, 32, 32)
    _y = torch.randint(0, 10, (256,))
    syn_ds = TensorDataset(_x, _y)
    train_loader = DataLoader(syn_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(syn_ds, batch_size=128, shuffle=False)
    log_ok("Synthetic data fallback: 256 samples, batch_size=128  ✓")


log_step("Training loop — 3 mini-batches")
try:
    model = MiniViT(image_size=32, patch_size=4, in_channels=3, num_classes=10,
                    embed_dim=128, depth=4, num_heads=4, mlp_dim=256,
                    dropout=0.1, emb_dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    model.train()
    for i, (images, labels) in enumerate(train_loader):
        if i >= 3:
            break
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        log_ok(f"  batch {i+1}: loss={loss.item():.4f}")
    log_ok("Training loop: 3 batches completed  ✓")
except Exception as e:
    log_error("Training loop", e)


log_step("Evaluation loop — 3 mini-batches")
try:
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            if i >= 3:
                break
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=-1)
            acc = preds.eq(labels).float().mean().item() * 100
            log_ok(f"  batch {i+1}: loss={loss.item():.4f}  acc={acc:.1f}%")
    log_ok("Evaluation loop: 3 batches completed  ✓")
except Exception as e:
    log_error("Evaluation loop", e)


log_step("LR Scheduler — SequentialLR (warmup + cosine)")
try:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=2
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=18, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[2]
    )
    lrs = []
    for _ in range(5):
        optimizer.step()   # must precede scheduler.step() (PyTorch >= 1.1)
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    log_ok(f"LR over 5 steps: {[f'{lr:.2e}' for lr in lrs]}  ✓")
except Exception as e:
    log_error("LR Scheduler", e)


log_step("Attention extraction for CLS visualisation")
try:
    model.eval()
    with torch.no_grad():
        # Use real test_dataset if available, else a random image
        if test_dataset is not None:
            raw_img = test_dataset[0][0]
        else:
            raw_img = torch.randn(3, 32, 32)
        sample = raw_img.unsqueeze(0).to(device)
        logits, attn_list = model(sample, return_attention=True)
        assert len(attn_list) == 4
        last_attn = attn_list[-1].squeeze(0).cpu()   # (4, 65, 65)
        cls_attn = last_attn[:, 0, 1:]               # (4, 64) — CLS row → patches
        cls_attn_grid = cls_attn.reshape(4, 8, 8)    # (heads, 8, 8)
        assert cls_attn_grid.shape == (4, 8, 8)
        log_ok(f"CLS attention grid: {cls_attn_grid.shape}  ✓")
except Exception as e:
    log_error("Attention extraction", e)


log_step("Positional embedding cosine similarity")
try:
    pos_emb = model.pos_embed[0, 1:].cpu()           # (64, 128)
    pos_norm = F.normalize(pos_emb, dim=-1)
    sim = pos_norm @ pos_norm.T                        # (64, 64)
    assert sim.shape == (64, 64)
    diag_mean = sim.diagonal().mean().item()
    assert abs(diag_mean - 1.0) < 1e-4, f"Diagonal should be 1.0, got {diag_mean}"
    log_ok(f"Similarity matrix {sim.shape}, diagonal mean={diag_mean:.4f}  ✓")
except Exception as e:
    log_error("Positional embedding similarity", e)


# ── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
if errors_log:
    print(f"\n  {len(errors_log)} ERROR(S) FOUND:\n")
    for i, err in enumerate(errors_log, 1):
        print(f"  {i}. [{err['step']}]\n     {err['error']}\n")
    sys.exit(1)
else:
    print("\n  ALL STEPS PASSED ✓\n")
    sys.exit(0)
