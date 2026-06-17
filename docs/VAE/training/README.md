# Training and Evaluating VAEs

A teaching-first series on how to **train** a Variational Autoencoder and, just
as importantly, how to **know whether it worked**.

The companion [VAE theory series](../README.md) (`VAE-01` through `VAE-09`)
answers *what a VAE is and why the math holds together*. This series answers a
different, more practical question: **given a dataset and a VAE, how do you take
it from random weights to a trained, trustworthy model — and how do you measure
"trustworthy"?**

You do not need to have read the whole theory series first. We recap every idea
we lean on as it comes up. If you know what a neural network is, what a loss
function is, and what gradient descent does, you have enough to start.

---

## What this series covers

The journey of training a model has five stages, and each chapter is one stage:

```mermaid
flowchart LR
    A["1 - Data<br/>what feeds the model"] --> B["2 - Model<br/>encoder + decoder"]
    B --> C["3 - Objective<br/>the loss"]
    C --> D["4 - Optimization<br/>the training loop"]
    D --> E["5 - Evaluation<br/>did it work?"]
```

| Chapter | Title | The question it answers |
|---------|-------|-------------------------|
| [01](01-introduction.md) | Introduction | What does "training a VAE" actually mean, end to end? |
| [01a](01a-richer-posteriors.md) | Richer posteriors *(optional aside)* | Why is the posterior a Gaussian — could it be a mixture, Gamma, or something richer? |
| [02](02-datasets.md) | Datasets | What does the training data look like — and do diffusion and flow-matching models want the same data? |
| [03](03-the-training-loop.md) | The training loop | How does the loop run, and how do I read its output and spot trouble? |
| [04](04-intrinsic-evaluation.md) | Intrinsic evaluation | Is the model good *on its own terms*? (And is FID the right metric here?) |
| [05](05-extrinsic-evaluation.md) | Extrinsic evaluation | Is the learned representation *useful for downstream tasks*? |
| [06](06-evaluation-protocol.md) | Evaluation protocol | Put it all together: a checklist and one fully worked example. |

---

## The running example

To keep things concrete, the whole series follows **one** small model: a
conditional VAE with a Negative-Binomial decoder (`CVAE_NB`) trained on a subset
of **PBMC** single-cell RNA-seq data — a few thousand immune cells, each
described by its gene-expression counts, with a known cell-type label we can use
later to test the model.

Don't worry if "Negative-Binomial decoder" or "PBMC" mean nothing yet — both are
introduced gently in chapters 01 and 02. The point is that every abstract idea
in this series is also shown happening to this one real model.

---

## Runnable companions

Reading explains; running convinces. Each chapter has runnable counterparts that
follow the same parallel layout used across the project:

- **Scripts**: `examples/vae/training/` — production-style, config-driven `.py`.
- **Notebooks**: `notebooks/vae/training/` — step-by-step `.ipynb` walkthroughs.

Everything is **size-configurable**. The exact same code runs as a fast
`smoke` test on your laptop (seconds, CPU — just to prove the workflow is
wired correctly) or as a `realistic` training run on a GPU pod (via `ops/`).
You change a config value, not the code. Chapter 03 explains this pattern in
full.

---

## Prerequisites and pointers

- If a piece of VAE theory feels too compressed here, the deeper derivation is
  in the theory series — we link to the specific chapter each time (for example,
  the ELBO in [VAE-02](../VAE-02-elbo.md), the reparameterization trick in
  [VAE-04](../VAE-04-reparameterization.md), and count decoders in
  [VAE-07](../VAE-07-NB-ZINB.md)).
- Start at [Chapter 01](01-introduction.md) and read in order; each chapter
  recaps what the previous one established before moving on.
