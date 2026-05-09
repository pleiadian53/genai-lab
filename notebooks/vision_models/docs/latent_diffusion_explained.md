# Latent Diffusion / Latent Flow Matching — Explainer

This doc covers what "latent of a pretrained VAE" means in the context of
modern image generation, why it works, and how it differs from JEPA.
Written as a tutorial companion to the perturbation-imaging V-series and
the (deferred) `latent_vae_concepts.ipynb`.

---

## The problem: pixel-space generation is wasteful

A 256×256 RGB image has 196,608 dimensions. Most of those dimensions
are perceptually redundant — neighboring pixels are highly correlated,
and small high-frequency details (texture noise, JPEG artifacts) are
not what humans (or downstream tasks) care about.

Training a diffusion model or flow-matching model directly in pixel
space means:

- The model spends capacity learning correlations that are not
  semantically meaningful
- Each training step touches all ~200K dimensions
- Sampling requires running the iterative ODE/SDE through that full
  pixel space at every step

Empirically: pixel-space diffusion at 256×256 needed ~150M parameter
models (DDPM, Improved-DDPM) and dozens of GPU-days to train at a
quality that latent diffusion now achieves in a fraction of the
compute. The wasted capacity is real.

The fix: **don't generate in pixel space. Generate in a compressed
"latent" space, then decode to pixels at the end.**

---

## The recipe in three steps

### Step 1 — Compress the image with a VAE

Train (or use a pretrained) Variational Autoencoder. The encoder $\varphi$
maps an image $x \in \mathbb{R}^{H \times W \times C}$ to a latent
$z \in \mathbb{R}^{h \times w \times c}$ where $h, w$ are typically
$H/8, W/8$ and $c$ is small (e.g., 4 for SD-VAE).

For a 256×256 RGB image:
$$
x \in \mathbb{R}^{256 \times 256 \times 3} \quad \xrightarrow{\;\varphi\;} \quad z \in \mathbb{R}^{32 \times 32 \times 4}
$$

That's a **48× reduction in dimensionality** (196,608 → 4,096) with most
of the perceptual information preserved.

The VAE is trained with three losses, balanced by hyperparameters:

- **Reconstruction**: $\|\psi(\varphi(x)) - x\|^2$ or perceptual (LPIPS)
- **KL regularization**: $\mathrm{KL}(q(z|x) \,\|\, \mathcal{N}(0, I))$ —
  keeps the latent space well-behaved (continuous, near-Gaussian)
- **Adversarial** (optional): a small discriminator on the decoder
  output, used in SD-VAE and VQ-GAN to prevent the blurriness that
  plain reconstruction loss tends to produce

The KL term is what makes this a *variational* autoencoder rather than
a vanilla autoencoder — it forces the latent to be smooth and Gaussian,
which is critical because the next step assumes you can interpolate and
sample in latent space.

### Step 2 — Train the generative model in latent space

This is where flow matching (or diffusion) lives. With the VAE frozen,
encode the entire training set once into latents $\{z^{(i)}\}$. Then
train a velocity field $v_\theta(z_t, t, c)$ on the conditional
flow-matching loss:

$$
\mathcal{L}(\theta) = \mathbb{E}_{t, z_0, z_1}\!\left[\|v_\theta(z_t, t, c) - u_t(z_0, z_1)\|^2\right]
$$

where $z_t = (1-t)z_0 + t z_1$, $z_0$ is Gaussian noise, $z_1$ is a
real latent, and $c$ is conditioning (perturbation embedding, class
label, text prompt — whatever's available).

The architecture is whatever fits image-shaped tensors: U-Net or DiT
on the spatial latent grid. For a 32×32×4 latent, this is much smaller
and faster than running the same backbone on a 256×256×3 pixel image.

### Step 3 — Decode at sampling time

To generate a new image:

1. Sample $z_0 \sim \mathcal{N}(0, I)$ in latent space
2. Integrate the ODE $\dot z_t = v_\theta(z_t, t, c)$ from $t=0$ to $t=1$
3. Decode: $\hat x = \psi(z_1)$

The VAE decoder $\psi$ handles the high-frequency detail; the
generative model only had to learn the low-frequency, semantically
meaningful structure. This division of labor is why latent diffusion
produces sharp, realistic images at far less training cost than
pixel-space diffusion.

---

## Why it works

The key insight, made explicit by Rombach et al. (2022, the original
Latent Diffusion paper): **most of the perceptually relevant information
in an image lives on a low-dimensional manifold, and a well-trained VAE
gives you a coordinate system for that manifold.**

Concretely, two things have to be true:

1. **The decoder is good enough.** $\psi(\varphi(x)) \approx x$ for real
   images $x$. If the VAE can't reconstruct, you can't generate well no
   matter how good the latent diffusion is.
2. **The latent is well-behaved.** Smooth, near-Gaussian, with
   semantically related images close to each other. The KL
   regularization is what enforces this; without it, the latent is just
   a dense codebook with arbitrary geometry.

When both hold, the generative model only needs to learn the *latent*
distribution — not the full pixel distribution. That's a far smaller
problem.

---

## Latent diffusion vs JEPA

JEPA (Joint Embedding Predictive Architecture, LeCun et al.) also uses
a latent space. The two are easy to confuse — they share substrate but
solve different problems.

| | Latent diffusion / flow matching | JEPA |
|---|----------------------------------|------|
| **Goal** | Generate novel data samples | Learn predictive representations |
| **Decoder back to pixels** | Yes — VAE decoder $\psi$ | **No** — predictions stay in latent space |
| **What the latent encodes** | A compressed pixel representation; decoder $\psi$ must reconstruct | A semantic representation; no reconstruction required |
| **Training of latent space** | Reconstruction + KL + (optional) adversarial — pixel-faithful | Predict target embedding from context embedding, with VICReg / SimSiam / Barlow Twins-style collapse prevention |
| **What you can do with the latent** | Sample new images via $\psi(z)$ | Embed inputs for downstream classification, retrieval, control |
| **Inference output** | Image (after $\psi$) | Embedding |

The deeper distinction: **latent diffusion's latent space is
pixel-faithful — every point in it must decode to a plausible image.
JEPA's latent space is task-faithful — every point in it must be useful
for predicting other latent points, but it has no obligation to
correspond to any particular pixel pattern.**

This has practical consequences:

- A latent diffusion model can be inverted (encode an image, edit the
  latent, decode) — used for image editing, inpainting, style transfer.
  JEPA cannot, because there's no decoder.
- A JEPA latent can drop pixel-level detail (e.g., texture, exact
  positions) in favor of semantic structure. A VAE latent cannot,
  because the decoder will need that detail to reconstruct.

### Common ground

Both approaches:
- Compress the input into a lower-dimensional latent space
- Decouple "what matters" from raw pixel statistics
- Make downstream tasks computationally tractable

And both are now standard substrates for downstream models. In a
modern stack you might see *both* in use: a JEPA-style encoder for
representation learning, and a VAE-decoder pair for generation. They
are complementary, not competing.

---

## Where they meet in genai-lab

The perturbation flagship plans to stack predictive and generative
components on the same latent substrate:

1. Encode raw scRNA-seq counts → CVAE_NB latent (autoencoder-style,
   reconstruction-faithful)
2. Predict perturbed-state latent via JEPA (predictive, no decoder
   needed at this stage)
3. Sample diverse plausible perturbed states from the JEPA latent via
   latent diffusion (generative; uses the CVAE decoder for the final
   step)

For the perturbation-imaging path specifically, the architecture is
narrower:

```
real image → multichannel VAE encoder → latent z
                                          │
                              perturbation embedding c
                                          │
                                 VelocityDiT2D
                                          │
                                          ▼
                                 latent ODE sampler
                                          │
                                          ▼
                          multichannel VAE decoder → image
```

JEPA isn't in this pipeline yet. A future extension — predict the
perturbed latent before generating it — would add a JEPA-style
predictive head between the VAE encoder and the DiT, mirroring the
scRNA-seq side.

---

## Honest caveats

Latent diffusion is not free of trade-offs:

1. **The VAE bottleneck.** Whatever the VAE can't reconstruct, the
   generative model can't learn either. SD-VAE on natural images
   loses fine text and high-frequency detail; this shows up as garbled
   text and texture artifacts in Stable Diffusion outputs. Pretrained
   VAEs on out-of-domain data (medical imaging, cell painting) lose
   even more.
2. **Two-stage training is harder to tune.** The VAE and the
   generative model have separate objectives; misaligned objectives
   (a VAE that throws away discriminative detail) hurt downstream
   sample quality in ways that are hard to diagnose.
3. **Latent geometry is not always Gaussian-shaped.** The KL term
   pushes toward $\mathcal{N}(0, I)$ but doesn't enforce it strictly.
   Diffusion / flow matching that *assumes* a Gaussian source
   distribution can be off when the latent's actual distribution
   isn't quite Gaussian.
4. **Continuous vs discrete latents have different trade-offs.**
   KL-VAE produces continuous latents (compatible with diffusion/flow
   matching out of the box). VQ-VAE produces discrete latents (sharper
   reconstructions, but requires Gumbel-softmax or similar to interface
   with continuous samplers).

The perturbation-imaging V-series uses a continuous (KL-VAE) latent for
V2 to keep V3+ uncomplicated. VQ-VAE is a possible V6 if reconstruction
quality limits sample fidelity.

---

## References

- Rombach et al. (2022), "High-Resolution Image Synthesis with Latent
  Diffusion Models," CVPR
- Kingma & Welling (2014), "Auto-Encoding Variational Bayes," ICLR —
  the original VAE
- van den Oord et al. (2017), "Neural Discrete Representation Learning"
  — VQ-VAE
- Lipman et al. (2022), "Flow Matching for Generative Modeling" — the
  flow-matching objective used here
- LeCun (2022), "A Path Towards Autonomous Machine Intelligence" —
  the JEPA framing
- Assran et al. (2023), "Self-Supervised Learning from Images with a
  Joint-Embedding Predictive Architecture" (I-JEPA)

---

## Related in this repo

- [`notebooks/vision_models/docs/connection_to_genai.md`](connection_to_genai.md)
  — why vision_models exists in this project
- [`examples/vision_models/perturbation_imaging/`](../../../examples/vision_models/perturbation_imaging/)
  — production application using these ideas
- [`dev/planning/perturbation_imaging_path.md`](../../../dev/planning/perturbation_imaging_path.md)
  — full path plan
- (deferred) `notebooks/vision_models/latent_vae_concepts.ipynb` —
  hands-on encode/decode/interpolate demo with SD-VAE on natural images
