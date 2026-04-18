# MNIST Flow Matching — Results & Analysis

Results from running [`01_mnist_flow_matching.py`](../01_mnist_flow_matching.py)
at production scale on an A40 pod. Companion to the notebook warmup
[`notebooks/flow_matching/02_flow_matching_mnist_warmup.ipynb`](../../../notebooks/flow_matching/02_flow_matching_mnist_warmup.ipynb),
which was run at small scale for pedagogy.

---

## TL;DR

A 300-epoch flow matching run with `base_channels=128` produces **high-quality
MNIST samples** — every digit cleanly legible, good class diversity, no mode
collapse. Training converges cleanly with a classic monotonic descent in loss,
but at epoch ~132 the validation MSE loss bottoms out at 0.159 and drifts upward
as training continues, while training loss keeps falling. **Despite this
"overfit signal" on val MSE, sample quality at epoch 300 is visibly better
than at mid-training** — a clean illustration of why MSE val loss is a weak
proxy for generative sample quality.

## Experimental Setup

```bash
python examples/flow_matching/01_mnist_flow_matching.py \
    --epochs 300 --batch-size 256 --base-channels 128 --n-steps 100 \
    --checkpoint-dir "$RUN_DIR"
```

| | |
|---|---|
| Model | U-Net velocity field, `base_channels=128`, multipliers (1, 2, 4), 2 res blocks per level, sinusoidal time embedding (128-dim) + FiLM |
| Parameters | ~52 M (scales ~16× from the 3.28 M warmup model which used `base_channels=32`) |
| Interpolant | Linear (rectified flow): $x_t = (1-t) x_0 + t\, x_1$, $u_t = x_1 - x_0$ |
| Loss | CFM (MSE between predicted and true velocity) |
| Optimizer | AdamW, lr = 1e-4 |
| Data | MNIST, batch size 256, 234 batches/epoch |
| Sampler (final) | Euler, 100 steps |
| Hardware | NVIDIA A40 48 GB, via RunPod + SkyPilot (`ops/provision_cluster.py`) |
| Wall time | 6 h 4 min (300 epochs × ~1 min 12 s/epoch) |
| Cost | ~$2.35 at $0.39/hr |
| Run directory | [`runs/flow_matching/mnist_fm_20260417_175639/`](../../../runs/flow_matching/mnist_fm_20260417_175639/) |

## Training Dynamics

| Epoch | Train | Val | Notes |
|------:|------:|------:|-------|
| 1     | 0.3110 | 0.2323 | Initial (train loss is batch-average over whole epoch; val is endpoint — see note at end) |
| 5     | 0.1877 | 0.1882 | Rapid early convergence |
| 10    | 0.1772 | 0.1744 | Train–val nearly tied |
| 20    | 0.1706 | 0.1708 | First plateau forming |
| 50    | 0.1644 | 0.1627 | Still descending |
| 100   | 0.1613 | 0.1614 | Approaching asymptote |
| **132** | **0.1596** | **0.1589** ← | **Best val; train and val essentially equal** |
| 150   | 0.1597 | 0.1618 | Val starts drifting up |
| 200   | 0.1569 | 0.1618 | |
| 250   | 0.1529 | 0.1620 | Gap widening |
| 300   | **0.1483** | 0.1695 | **Final — train–val gap +0.021** |

Two complementary views:

```
Train:  0.31 ────▼──── 0.17 ── 0.16 ── 0.16 ── 0.15 (still descending)
Val:    0.23 ────▼──── 0.17 ── 0.16 ── 0.16 ── 0.17 (bottomed, drifted up)
                                       ▲
                                  epoch 132 minimum
```

**The divergence is remarkably symmetric**: between epochs 132 and 300, train
loss dropped −0.0113 and val loss rose +0.0106. This is the classic
"memorization" signature — the model gets better at fitting the specific
training trajectories but marginally worse at predicting held-out trajectories.

## Sample Quality (Epoch 300)

The end-of-training sample grid from the final checkpoint, 100-step Euler
integration:

![MNIST samples from epoch 300](../../../runs/flow_matching/mnist_fm_20260417_175639/samples_final.png)

Every digit class is cleanly recognizable. Strokes are crisp and continuous,
backgrounds are pure black, proportions are correct. No mode collapse — all 10
digits appear with roughly uniform frequency in the 8×8 grid. Compared to the
30-epoch `base_channels=32` warmup notebook, these samples are dramatically
sharper and more legible.

## Key Finding

**MSE val loss bottomed at epoch 132, but samples at epoch 300 are visibly
better.** This is a clean empirical demonstration of a general principle in
generative modeling: *the training objective is a proxy for the real goal
(sample quality), and at some point they decouple.* The later stages of
training sharpen the velocity field in ways that help generation but that
look like "overfitting" through the narrow lens of held-out MSE.

Practical implication: **don't pick generative model checkpoints by val loss
alone.** At minimum:

1. Visually inspect samples from the val-minimum checkpoint and the final
2. If possible, compute a quantitative sample-quality metric (see below) on both

## Next Steps

- [ ] Generate samples from `checkpoint_epoch0130.pt` and A/B against
      `model_final.pt` to confirm the "samples keep improving past val minimum"
      hypothesis quantitatively
- [ ] Compute classifier-based quality score (Section B.1 below) on both
      checkpoints — the first quantitative number for this project
- [ ] Step-count ablation at scale: repeat the warmup-notebook ablation (1,
      5, 10, 20, 50, 100 steps) with the final model to confirm rectified
      flow's few-step advantage holds at larger model size
- [ ] W&B integration is now in place — next run should be launched with
      `--wandb` to get live training curves instead of post-hoc log parsing
- [ ] Add this run to [`output/flow_matching/`](../../../output/flow_matching/)
      once a second run exists to compare against

---

# Appendix: Interpreting These Results

Two questions that come up when working with generative models,
addressed from what this experiment showed.

## A. How do you assess sample quality without ground truth?

Visual inspection is fast, honest, and often sufficient — but it's subjective
and doesn't scale to sweeps or commit checks. Several quantitative metrics
complement it; none is perfect, so the practical answer is to use **two or
three together plus occasional visual inspection**.

### A.1 Classifier-based evaluation (recommended starting point for MNIST)

Train a classifier on real MNIST (trivially 99%+ accurate with a small CNN),
then run it on generated samples. Report two numbers:

- **Mean confidence** on samples (quality) — high values mean the classifier
  is "certain" about what digit each sample is → samples look like real
  digits.
- **Class distribution** over many samples (diversity) — should be close to
  the training prior (roughly uniform 1/10 for MNIST). Skew indicates mode
  collapse.

Cheap to compute, directly interpretable, and the classifier can be reused
across runs. For MNIST specifically this is the single most useful metric.

Downside: assumes the classifier is perfect, which it isn't — if your
generative model produces a systematic artifact the classifier happens to be
blind to, you won't see it in this metric. Always backstop with visual
inspection.

### A.2 FID (Fréchet Inception Distance)

The standard for natural-image generation. Uses Inception v3 features
(penultimate layer), models real and generated feature distributions as
multivariate Gaussians, computes the Fréchet distance between them.

- **Pros**: Captures both quality and diversity in one number; widely reported
  so you can compare with papers.
- **Cons**: Inception v3 was trained on ImageNet and is a poor feature
  extractor for MNIST (grayscale, 28×28, no natural-image statistics). For
  MNIST specifically, use a MNIST-trained classifier's penultimate layer
  instead — same math, better-calibrated features.
- **Interpretation**: Lower is better; requires thousands of samples for
  stable estimates.

### A.3 Precision / Recall for generative models (Kynkäänniemi et al. 2019)

Decouples "are my samples on-manifold?" (precision, quality) from "do they
cover the real data manifold?" (recall, diversity). Computed from k-NN
graphs in feature space.

- **Pros**: Diagnostic — you can tell *why* a model underperforms (bad
  precision = blurry/wrong samples; bad recall = mode collapse).
- **Cons**: Requires tuning k; features matter as much as for FID.

### A.4 Density estimation metrics (when tractable)

For flow models specifically, the continuous change-of-variables formula lets
you compute **exact log-likelihood** of held-out data under the learned flow.
Report as bits-per-dimension (BPD) to compare across image sizes.

- Requires integrating the trace of the velocity Jacobian along the ODE
  trajectory, which is expensive but feasible.
- Not implemented in the current example script; would be a clean add.

### A.5 Sample-efficient diagnostics

Cheap sanity checks that don't need large sample counts:

- **Per-class sample counts** from the classifier above — immediate mode-
  collapse detector
- **Pixel-mean and variance** comparison (real vs generated) — catches gross
  miscalibration
- **Fixed-noise sampling across checkpoints** — if samples *at the same noise
  seed* look increasingly recognizable as training progresses, that's a cheap
  trajectory diagnostic (what Section 7 of the warmup notebook does visually)

### When visual + metrics disagree

Metrics sometimes miss what your eyes catch, and vice versa. Common failure
modes:

- **Metrics say good, samples look bad**: the model may be producing samples
  that match feature statistics but fail high-level coherence (e.g., samples
  that a classifier confidently classifies as "3" but that humans see as
  malformed)
- **Samples look fine, metrics say bad**: often a feature-extractor mismatch
  (FID with ImageNet features on a non-ImageNet dataset) or a too-small
  sample pool

Rule of thumb: **trust the metric when it tracks visual assessment across
many runs**. Start with visual for the first few runs, gradually shift weight
to metrics once they prove predictive.

---

## B. Why velocity-field prediction behaves differently from "conventional" supervised learning

Flow matching looks superficially like regression — you have inputs
$(x_t, t)$ and targets $u_t$, you minimize MSE. But the target structure
differs in ways that change the familiar supervised-learning intuitions.

### B.1 The target is **stochastic**, not deterministic

In classification, a given image $x$ has a unique label $y$. Flow matching's
training samples are generated by:

```
sample x0 ∼ p_data     (real MNIST digit, a specific one)
sample x1 ∼ N(0, I)    (a specific noise tensor)
sample t  ∼ U[0, 1]
compute x_t = (1-t)·x0 + t·x1  (the input)
set     u_t = x1 - x0          (the target)
```

Crucially, **many different $(x_0, x_1)$ pairs can produce nearly identical
$x_t$** — especially when $t$ is near 0.5 and $x_t$ is in the "between data
and noise" region. These different pairs have **different $u_t$ targets**.

**Consequence**: the Bayes-optimal predictor is the *conditional expectation*
$\mathbb{E}[u_t \mid x_t, t]$, and the irreducible loss is the conditional
variance $\text{Var}(u_t \mid x_t, t)$, which is **not zero**.

In a classifier, the optimal accuracy approaches 100% on clean data. In flow
matching, the optimal MSE approaches a positive number determined by the
geometry of the data-vs-noise interpolation. **A plateaued CFM loss at 0.15
does not mean the model is undertrained** — it may be at or near the Bayes
floor.

### B.2 Higher batch variance from **two** independent randomness sources

A classification batch has one random draw — the sample index. A flow
matching batch has three:

1. The data sample $x_0$
2. The noise sample $x_1$ (fresh every batch, even when you re-use $x_0$)
3. The time $t$ (uniform in $[0,1]$ each batch)

The per-batch loss can swing noticeably from one batch to the next even on
late-in-training weights, because you're evaluating the model at different
points along the noise-to-data continuum in each batch. This is why you see:

- Higher epoch-to-epoch variance in val loss than in a classifier (~0.001–
  0.005 is typical at convergence)
- Higher tqdm per-batch loss variance during training
- A larger apparent "effective batch size" needed for stable training —
  batch size 256 is on the low end; 512 or 1024 smooths things noticeably

### B.3 Train–val gap **doesn't map to memorization the same way**

In a classifier:
- Widening train–val gap (train ↓, val ↑) = the model is memorizing specific
  training inputs. Bad.

In flow matching:
- Widening train–val gap can mean the model is getting better at predicting
  velocities for the specific noise-data pairs seen in training. But the
  trajectories that matter for *generation* start from fresh $N(0,I)$ noise
  not seen in training — so the "memorization" of training trajectories
  doesn't directly degrade sample quality.
- This is exactly what we see in this run: train–val gap opens from 0 (epoch
  132) to 0.021 (epoch 300), yet samples at epoch 300 are visually better.

The val-loss-is-MSE view is measuring "how well does the model predict
velocities on held-out $(x_0, x_1)$ pairs" — a fair question, but one step
removed from the thing you actually care about ("how good are my samples").

### B.4 Val loss is a **weak proxy** for the real objective

In a classifier, val accuracy ≈ test accuracy ≈ the thing you deploy for.
There's a tight semantic connection.

In flow matching, the training loss is chosen for **tractability**, not
because it directly measures sample quality. The CFM objective is an
unbiased estimator of the score-matching objective (marginally, in
expectation) — but a low CFM loss doesn't *guarantee* good samples, and
vice versa.

This is why the community uses FID / IS / precision-recall as the actual
quality metric and treats the training loss as a proxy.

### B.5 Effective optimization has more **noise**, less signal per step

Because each batch samples independently from $t \in [0,1]$, your gradient
estimate at each step is an average of:
- 256 estimates of "velocity prediction at t ≈ 0.03"
- 256 estimates of "velocity prediction at t ≈ 0.71"
- …one estimate each at 256 different $t$ values

You're **not** averaging 256 independent estimates of the same quantity
(like you would in a classifier with fixed-type inputs). You're averaging
256 estimates of related-but-different quantities. The effective noise in
the gradient is higher, which is one reason flow matching / diffusion
training typically uses:
- Larger batch sizes than classifiers
- Longer training (in epochs) for a given model size
- EMA of weights for sampling (to smooth the effective model)

### B.6 What *does* look like conventional supervised learning

For completeness — some behaviors are actually classifier-like:

- **Loss monotonically decreases in expectation** (modulo batch noise)
- **Standard optimizers work**: AdamW, cosine LR schedule, gradient clipping
- **Wider / deeper models fit better** on training data
- **Batch size scales the usual way** (linear-scale the LR with batch size)

The novelty is mostly in *interpreting* the loss and the train–val gap,
not in how you train.

---

## Note on the "train is epoch-average, val is epoch-end" asymmetry

In the table above, epoch 1 shows train=0.31 and val=0.23 — the val loss
looks lower than train. This is mechanical, not a sign of something weird:

- **Train loss** is the running average over *all batches* in the epoch. Early
  batches have higher loss (weights are still being learned), later batches
  have lower loss. The average reflects both.
- **Val loss** is a single pass over the val set at the *end* of the epoch,
  using the *current* (best) weights.

So early in training, val-below-train is expected and doesn't mean val is
"easier". Late in training when loss is near-plateau, within-epoch weight
changes are small and the two numbers converge to being comparable. This
is the default behavior in essentially every ML framework (PyTorch Lightning,
Keras, HuggingFace Trainer, custom loops).

Not unique to flow matching, but worth keeping in mind when reading the
epoch-1 row.

---

## Related

- [`notebooks/flow_matching/02_flow_matching_mnist_warmup.ipynb`](../../../notebooks/flow_matching/02_flow_matching_mnist_warmup.ipynb) — pedagogical warmup
  with `base_channels=32`, 30 epochs, step-count ablation
- [`examples/docs/running_experiments.md`](../../docs/running_experiments.md) — the nohup pattern used
  to launch this run
- [`examples/docs/experiment_tracking.md`](../../docs/experiment_tracking.md) — W&B integration (not
  used for this run; next run will use `--wandb`)
- [`runs/README.md`](../../../runs/README.md) — artifact directory convention
- [`docs/flow_matching/`](../../../docs/flow_matching/) — flow matching theory docs
