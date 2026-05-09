# Troubleshooting Guide: mini_vit_from_scratch.ipynb

## Summary

Two issues were found and fixed when running the notebook and its test script
end-to-end on macOS with Python 3.12 / PyTorch 2.9.1 (MPS device).

| # | Location | Issue | Fix |
|---|---|---|---|
| 1 | Notebook Cell 6 | `HTTPError` downloading CIFAR-10 | One-time manual download; data now cached |
| 2 | `test_mini_vit_e2e.py` | `num_workers=2` crashes on macOS (`spawn`) | Changed to `num_workers=0` in test script |

---

## Error 1: HTTPError When Downloading CIFAR-10 (Notebook Cell 6)

### Symptom

```
HTTPError: HTTP Error 403: Forbidden
```

Raised inside `torchvision.datasets.CIFAR10(..., download=True, ...)`.

### Location

- **File**: `mini_vit_from_scratch.ipynb`
- **Cell**: Cell 6 (patch visualisation) and Cell 22 (data loading for training)

### Root Cause

torchvision downloads CIFAR-10 from `https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz`.
The remote server intermittently returns HTTP 403. This is a transient server-side issue,
not a code bug. Once the 170 MB archive is present locally, `download=True` is a no-op.

### Solution

Run the test script once (or any cell with `download=True`) while the server is reachable.
After a successful download the data lives at:

```
notebooks/vision_models/data/cifar-10-batches-py/
notebooks/vision_models/data/cifar-10-python.tar.gz
```

The `root="./data"` path in the notebook resolves relative to the Jupyter working directory,
which must be `notebooks/vision_models/` for the data to be found without re-downloading.

**Alternative: download manually**

```bash
mkdir -p notebooks/vision_models/data
curl -o notebooks/vision_models/data/cifar-10-python.tar.gz \
     https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzf notebooks/vision_models/data/cifar-10-python.tar.gz \
     -C notebooks/vision_models/data/
```

No notebook code change is required — `torchvision` will find the extracted directory
and skip downloading.

---

## Error 2: `num_workers=2` Crashes in Test Script on macOS

### Symptom

```
An attempt has been made to start a new process before the current process
has finished its bootstrapping phase.

This probably means that you are not using fork to start your child
processes and you have forgotten to use the proper idiom in the main module:

    if __name__ == '__main__':
        freeze_support()
        ...
```

Raised in the `Training loop` and `Evaluation loop` test steps.

### Location

- **File**: `test_mini_vit_e2e.py`
- **Steps**: `Training loop — 3 mini-batches`, `Evaluation loop — 3 mini-batches`

### Root Cause

Python 3.8 changed the default multiprocessing start method on macOS from `fork` to
`spawn`. When `DataLoader` is created with `num_workers > 0`, it spawns worker
processes. The `spawn` method requires the entire script to be re-importable under
`if __name__ == '__main__'`, which a flat test script is not.

**This does NOT affect the notebook itself.** Jupyter runs in an already-initialised
main process context, so `num_workers=2` works correctly there.

### Solution

**Before (incorrect in a `.py` script on macOS):**

```python
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=2, pin_memory=pin)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                          num_workers=2, pin_memory=pin)
```

**After (correct for test scripts on macOS):**

```python
# num_workers=0 required in a plain .py script on macOS (spawn start method).
# num_workers>0 works fine in Jupyter because the notebook is already in a
# proper main process context.
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=0, pin_memory=pin)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                          num_workers=0, pin_memory=pin)
```

Alternatively, the entire test script body could be wrapped in
`if __name__ == '__main__': ...`, but `num_workers=0` is simpler and sufficient
for a test script where throughput is not the goal.

---

## Warning: LR Scheduler Step Order (Non-Fatal)

### Symptom

```
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`.
In PyTorch 1.1.0 and later, you should call them in the opposite order.
```

### Location

- **File**: `test_mini_vit_e2e.py`
- **Step**: `LR Scheduler — SequentialLR (warmup + cosine)`

### Root Cause

The isolated scheduler unit test called `scheduler.step()` in a loop without first
calling `optimizer.step()`. PyTorch emits a warning because this pattern was the
pre-1.1 API and can skip the first LR value.

### Solution

Add a dummy `optimizer.step()` before each `scheduler.step()` in the unit test:

```python
for _ in range(5):
    optimizer.step()   # must precede scheduler.step() (PyTorch >= 1.1)
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])
```

This is not an issue in the notebook's actual training loop, where `scheduler.step()`
is correctly called at the end of each epoch after `optimizer.step()` inside
`train_one_epoch()`.

---

## Quick Fix Summary

| File | Change |
|---|---|
| `test_mini_vit_e2e.py` | `num_workers=2` → `num_workers=0` in both DataLoaders |
| `test_mini_vit_e2e.py` | Add `optimizer.step()` before `scheduler.step()` in LR test |
| CIFAR-10 data | Download once; lives at `notebooks/vision_models/data/` |

---

## End-to-End Test

```bash
cd notebooks/vision_models
conda activate genailab
python test_mini_vit_e2e.py
```

Expected output (abridged):

```
  STEP: Device setup         ✓ device = mps
  STEP: PatchEmbedding       ✓ (2,3,32,32) → torch.Size([2, 64, 128])
  STEP: MHSA                 ✓ out torch.Size([2, 65, 128]), row_sum=1.0000
  STEP: MLP                  ✓ (2,65,128) → torch.Size([2, 65, 128])
  STEP: TransformerBlock     ✓ out torch.Size([2, 65, 128])
  STEP: MiniViT forward      ✓ logits torch.Size([4, 10]), 4 attention maps
  STEP: MiniViT on device    ✓ Forward pass on mps
  STEP: Data loading         ✓ CIFAR-10 train=50,000  test=10,000
  STEP: Training loop        ✓ 3 batches completed
  STEP: Evaluation loop      ✓ 3 batches completed
  STEP: LR Scheduler         ✓ warmup + cosine LR values confirmed
  STEP: Attention extraction ✓ CLS attention grid: torch.Size([4, 8, 8])
  STEP: Positional embedding ✓ Similarity matrix torch.Size([64, 64])

  ALL STEPS PASSED ✓
```

---

**Document created**: 2026-05-05  
**Tested with**: torch==2.9.1, torchvision==0.20+, Python 3.12, macOS (MPS)
