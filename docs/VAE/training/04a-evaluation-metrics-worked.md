# Chapter 04a — Evaluation Metrics, Worked: From Computer Vision to Cells

*An optional deep-dive aside for [Chapter 04](04-intrinsic-evaluation.md). The main
chapter named the metrics; here we open the hood on each one with real numbers —
first the computer-vision version, then its translation to gene expression.
Skippable; [Chapter 05](05-extrinsic-evaluation.md) doesn't depend on it. Symbols
are in the [notation reference](notation.md).*

Chapter 04 surveyed the generative-evaluation landscape and argued, in particular,
that FID doesn't transfer to cells *as defined* but its underlying idea does. That
argument is worth making for *every* metric in the survey, not just FID — and the
clearest way to make it is to actually compute each one on a toy example small
enough to follow by hand, then ask what changes when the data points become cells
instead of images.

Keep one framing in mind throughout (it's the recap from Chapter 04). Almost every
metric is chasing **fidelity** (is each sample realistic?), **diversity/coverage**
(do the samples span the whole distribution, or mode-collapse to a few?), or
**likelihood** (does the model assign high probability to held-out data?). As we
work each metric, notice which of the three it actually measures — that, more than
the formula, is what tells you whether you even *want* it for your problem.

## Inception Score: confidence and variety, through a classifier

The **Inception Score (IS)** scores a *set* of generated images using a pretrained
classifier — InceptionV3 — and rewards two things at once: each image should be
classified *confidently* (a proxy for fidelity), while the set as a whole should
use *many* classes (a proxy for diversity). Writing $p(y \mid x)$ for the
classifier's distribution over labels $y$ given a generated image $x$, and $p(y)$
for the average of that over all generated images (the marginal label
distribution), the score is

$$
\text{IS} = \exp\left( \mathbb{E}_{x}\left[ \text{KL}\left( p(y \mid x) \Vert p(y) \right) \right] \right)
$$

The KL inside is large when each image's label distribution $p(y \mid x)$ is peaky
(confident) *and* far from the flat marginal $p(y)$ (which only happens if
different images pick different classes). Here $\mathbb{E}_x$ averages over
generated images and the outer $\exp$ is cosmetic, putting the score on a friendlier
scale.

Work it on three classes and two generated images. Say the classifier returns
$p(y \mid A) = (0.9, 0.05, 0.05)$ — confidently class 1 — and
$p(y \mid B) = (0.05, 0.9, 0.05)$ — confidently class 2. The marginal is their
average, $p(y) = (0.475, 0.475, 0.05)$. The KL for image $A$ is

$$
0.9 \log\frac{0.9}{0.475} + 0.05 \log\frac{0.05}{0.475} + 0.05 \log\frac{0.05}{0.05} = 0.575 - 0.113 + 0 = 0.463
$$

and by symmetry image $B$ gives the same 0.463, so the mean KL is 0.463 and
$\text{IS} = \exp(0.463) \approx 1.59$. With three classes the *maximum* possible
score is 3 (perfectly confident *and* perfectly spread across all three classes);
our 1.59 reflects that the model only ever produced two of the three. Notice the
metric silently rewards diversity — if both images had been class 1, $p(y)$ would
equal each $p(y \mid x)$, every KL would be zero, and IS would bottom out at 1.

**Translating to cells.** IS is welded to a classifier, and for images that
classifier is the universally-available InceptionV3. For cells there is no such
canonical thing — you'd have to supply your own trusted **cell-type classifier**
$C(y \mid x)$ and compute $\exp(\mathbb{E}[\text{KL}(C(y \mid x) \Vert C(y))])$
over generated cells. That's well-defined, but it inherits all the biases and
blind spots of whatever classifier you picked, and there's no community-standard
choice to make scores comparable across papers. So an IS-analog is *possible* for
cells but rarely used — the metric's whole appeal in vision was the free, shared
classifier, and that's exactly what biology lacks.

## Fréchet distance: matching two clouds of features

Chapter 04 gave FID's formula; here we actually evaluate it, in one dimension where
the matrix algebra collapses to arithmetic. The Fréchet distance between two
Gaussians, real $\mathcal{N}(\mu_r, \Sigma_r)$ and generated
$\mathcal{N}(\mu_g, \Sigma_g)$, is
$\lVert \mu_r - \mu_g \rVert^2 + \mathrm{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$.
In one dimension the covariances become plain variances and the whole second term
simplifies to $(\sigma_r - \sigma_g)^2$, leaving

$$
\text{FD} = (\mu_r - \mu_g)^2 + (\sigma_r - \sigma_g)^2
$$

Suppose the real features have mean $\mu_r = 2$ and variance $\sigma_r^2 = 1$ (so
$\sigma_r = 1$), and the generated features have mean $\mu_g = 2.5$ and variance
$\sigma_g^2 = 1.5$ (so $\sigma_g \approx 1.225$). Then
$\text{FD} = (2 - 2.5)^2 + (1 - 1.225)^2 = 0.25 + 0.051 = 0.30$. The two terms read
cleanly: the first penalizes a *shifted* center (the generated cloud sits half a
unit too high), the second penalizes a *mismatched spread* (the generated cloud is
a touch too wide). Zero requires matching both.

**Translating to cells.** Nothing in that calculation needed images — it needed a
*feature space* in which to take means and covariances. For images that space is
InceptionV3's activations; for cells we simply choose a biology-appropriate space,
most cheaply the top principal components of the expression matrix. Project both
real and generated cells onto, say, the first 50 PCs, compute the mean vector and
covariance of each cloud, and plug into the full multivariate formula. The result
is a "Fréchet distance in PCA space" that means for cells exactly what FID means for
images — same center, same spread — without borrowing an image network. (Swap the
PCs for a single-cell foundation-model embedding and you get a richer, more
semantic version of the same metric.)

## The general pattern: a feature space (and a distribution) per modality

Step back and FID dissolves into a two-slot recipe: pick a **feature space** to look
at the data through, summarize each side (real and generated) as a **distribution**
in that space, and measure the distance between them. FID fills the first slot with
InceptionV3 and the second with a Gaussian — but both are modality-dependent choices,
and every data type that does generative modeling has quietly made its own. The
NB-versus-Gaussian point from [Chapter 04](04-intrinsic-evaluation.md) was only the
gene-count instance of a much broader idea.

The feature-space slot is where the modality shows up most visibly. The extractor has
to be pretrained *on that kind of data*, so that closeness in its feature space tracks
*meaningful* similarity rather than raw sample-by-sample difference — which is exactly
why InceptionV3 can't be reused on a waveform or a molecule. So each field grew its own
Fréchet metric on its own "Inception":

| Modality | Feature extractor (the "Inception") | Named metric |
|----------|-------------------------------------|--------------|
| Images | InceptionV3 (ImageNet) | FID — Fréchet Inception Distance |
| Audio | VGGish / PANNs (AudioSet) | FAD — Fréchet Audio Distance |
| Video | I3D (Kinetics) | FVD — Fréchet Video Distance |
| Molecules (SMILES) | ChemNet | FCD — Fréchet ChemNet Distance |
| Text | BERT-style embeddings | Fréchet BERT Distance; MAUVE (related) |
| Time series | TS2Vec / a pretrained sequence encoder | Context-FID |
| Single-cell counts | PCA, or an scRNA foundation model | Fréchet-in-PCA (our analog) |

These are all the *same metric* wearing a modality-appropriate lens — FID, FAD, FVD,
and FCD are literally the same Fréchet-distance formula computed on different
pretrained features. (FCD, the molecular one, is worth flagging for us: it's standard
in drug discovery for scoring generated molecules, a close cousin of the
gene-expression generation problem.)

The distribution slot is the subtler one, and it's where your gene-count instinct
generalizes. When you compare *inside a learned embedding* — InceptionV3's
activations, VGGish's, ChemNet's — a Gaussian fit is usually fine, because deep
embeddings tend to come out roughly bell-shaped no matter what the raw data looked
like; the network has, in effect, Gaussianized it. But if you compare in the *raw
data space*, with no embedding to launder the geometry, then the data's own shape
dictates the right family: gene counts want a **Negative-Binomial**, strictly
positive continuous quantities a **Gamma** or **log-normal**, bounded fractions a
**Beta**. And if you'd rather assume nothing at all about the shape, you drop the
parametric family entirely for a non-parametric distance like **MMD** (next section),
which compares the two distributions through a kernel without ever fitting a Gaussian
— or an NB, or anything else.

So the lesson is bigger than "NB instead of Gaussian for counts." An FID-like metric
is a *template* with modality-shaped holes — a feature space and a distribution — and
good evaluation means filling both to match the data in front of you, whether that's
pixels, waveforms, molecules, or cells. Text pushes the point one slot further: MAUVE
compares two text distributions in embedding space but through a KL-divergence
frontier rather than a Fréchet distance, a reminder that even the *third* slot — how
you measure the gap once you've chosen a space and a shape — is negotiable too.

## MMD and KID: comparing distributions without a model of them

FID assumed both feature clouds were Gaussian. **Maximum Mean Discrepancy (MMD)**
drops that assumption: it compares two *sets of samples* directly through a kernel
$k(a, b)$ that measures similarity between two points, and it is zero exactly when
the two distributions match. For a real set $\{x_i\}_{i=1}^m$ and a generated set
$\{y_j\}_{j=1}^n$ the (biased) estimate is

$$
\widehat{\text{MMD}}^2 = \frac{1}{m^2}\sum_{i,j} k(x_i, x_j) + \frac{1}{n^2}\sum_{i,j} k(y_i, y_j) - \frac{2}{mn}\sum_{i,j} k(x_i, y_j)
$$

In words: average within-real similarity, plus average within-generated similarity,
minus twice the cross similarity. If the sets are drawn from the same distribution,
the cross term matches the within terms and the whole thing cancels to (near) zero;
if they're different, the within-similarities outweigh the cross-similarity and it's
positive. (**KID**, the Kernel Inception Distance, is just this quantity computed on
InceptionV3 features — MMD wearing FID's clothes.)

Work it with the common **RBF kernel**
$k(a, b) = \exp(-\frac{(a-b)^2}{2\sigma^2})$, bandwidth $\sigma^2 = 1$, on tiny
1-D sets: real $X = \{0, 1\}$, generated $Y = \{0.5, 2\}$. The within-real block
averages $k(0,0){=}1$, $k(0,1){=}k(1,0){=}0.607$, $k(1,1){=}1$ to
$3.213 / 4 = 0.803$. The within-generated block averages $k(0.5,0.5){=}1$,
$k(0.5,2){=}k(2,0.5){=}0.325$, $k(2,2){=}1$ to $2.649 / 4 = 0.662$. The cross block
averages $k(0,0.5){=}0.882$, $k(0,2){=}0.135$, $k(1,0.5){=}0.882$, $k(1,2){=}0.607$
to $2.507 / 4 = 0.627$, doubled to $1.253$. So
$\widehat{\text{MMD}}^2 = 0.803 + 0.662 - 1.253 = 0.21$, and
$\widehat{\text{MMD}} \approx 0.46$ — clearly nonzero, correctly flagging that
$\{0.5, 2\}$ is not drawn from the same distribution as $\{0, 1\}$ (the outlier
at 2 is what drives it).

**Translating to cells.** This is the easy transfer, and it's why MMD is a favorite
for single-cell work: it needs *no feature extractor at all*, only a kernel and the
data points. Run it directly on expression vectors (or, for stability in high
dimensions, on PCA scores) and you have a principled real-versus-generated distance
with none of FID's image baggage. It's the metric that survives the trip to biology
most intact.

## Precision and recall: splitting fidelity from coverage

Every metric so far returns *one* number, which blurs two very different failures: a
model can generate gorgeous but repetitive samples (high fidelity, low diversity —
mode collapse), or varied but sloppy ones (high diversity, low fidelity).
**Precision and recall for generative models** separates them. Estimate the real
data's *manifold* as the union of small balls, one around each real point, each ball
reaching to that point's $k$-th nearest real neighbor; do the same for the generated
points. Then **precision** is the fraction of *generated* points that land inside the
*real* manifold (how many fakes are realistic), and **recall** is the fraction of
*real* points inside the *generated* manifold (how much of the real variety the
model reaches).

A 1-D example with $k = 1$ makes it concrete. Real points $R = \{0, 1, 2\}$, each
with its nearest neighbor a distance 1 away, give a real manifold of
$[0{\pm}1] \cup [1{\pm}1] \cup [2{\pm}1] = [-1, 3]$. Generated points
$G = \{1, 5\}$ are each other's nearest neighbor at distance 4, giving a generated
manifold of $[1{\pm}4] \cup [5{\pm}4] = [-3, 9]$. Now read off the two numbers. Of
the generated points, $1 \in [-1,3]$ but $5 \notin [-1,3]$, so **precision** is
$1/2 = 0.5$ — half the generated samples are unrealistic (the lone "5"). Of the
real points, all of $0, 1, 2$ fall in $[-3, 9]$, so **recall** is $3/3 = 1.0$ — the
model covers every real mode. That pair, 0.5 and 1.0, tells a story no single
number could: *the model is comprehensive but imprecise.*

**Translating to cells.** Precision and recall are purely geometric — nearest
neighbors and distances — so they move to expression or PCA space unchanged, and
they're especially valuable in biology because *mode collapse* is a real and
dangerous failure: a perturbation model that only ever generates the most common
response (high precision, low recall) would look fine on a per-gene mean check yet
be useless for discovery. Precision/recall catches exactly that.

## Likelihood scores: bits-per-dimension, perplexity, and the ELBO

The vision and language worlds also report *likelihood* — how probable real
held-out data is under the model — as **bits-per-dimension** for images (the
negative log-likelihood per pixel, converted to bits) or **perplexity** for
language. The unit conversion is just a change of logarithm base: one bit is
$\ln 2 \approx 0.693$ nats, so a per-element NLL of, say, 0.693 nats is exactly 1
bit. For a VAE we usually can't compute the exact likelihood, so the honest stand-in
is the held-out **ELBO**, tightened by the importance-weighted **IWAE** bound from
[Chapter 04](04-intrinsic-evaluation.md), reported in nats per cell.

**Translating to cells.** This transfers in spirit but not in units. "Bits per
pixel" is a natural image convention because pixels are interchangeable; "bits per
gene" is rarely quoted because genes are not. So in practice the comp-bio version is
simply the held-out IWAE (nats per cell): higher (less negative) means the model
assigns more probability to real unseen cells. Same question — how well does the
model explain held-out data — different customary units.

## The metric that's native to biology

One check has no computer-vision parent because it falls out of the data type
itself: **per-gene summary statistics**. Generate a batch of cells, then compare,
gene by gene, the real and generated means (and variances). Tiny worked version:
across three genes the real means are $(5, 2, 0.5)$ and the generated means are
$(4.8, 2.3, 0.4)$; the Pearson correlation between these two vectors is about
$0.99$, i.e. the generator reproduces which genes are highly versus lowly expressed.
Repeating for variances checks the *overdispersion* the NB decoder exists to
capture. It's cheap, interpretable, and the first thing a biologist will ask to see
— a reminder that the best metric is often the one native to the domain, not the one
imported from another.

## The bridge, at a glance

| CV metric | What it instantiates | Comp-bio analog | Transfers cleanly? |
|-----------|----------------------|-----------------|--------------------|
| Inception Score | classifier confidence + variety | needs a trusted cell-type classifier | rarely — no canonical classifier |
| FID | Fréchet distance on InceptionV3 features | Fréchet distance in PCA / embedding space | idea yes; swap the feature space |
| KID / MMD | distribution distance via a kernel | MMD directly on expression or PCA | yes — no feature extractor needed |
| Precision & Recall | k-NN manifolds in feature space | k-NN manifolds in expression / PCA | yes — purely geometric |
| Bits-per-dim / perplexity | per-element held-out likelihood | held-out IWAE (nats per cell) | yes — different units |
| — | (no image parent) | per-gene mean / variance agreement | native to biology |

## Recap, and back to the main path

Walking each metric by hand surfaces the same lesson five times. A generative metric
is really a choice of *what to measure* (fidelity, coverage, or likelihood), *in what
space* to measure it, and *with what distribution or distance* — and a Fréchet-style
metric exposes all three as modality-shaped slots. The computer-vision instances bake
in an image-specific space (InceptionV3) and a Gaussian, and *those* fillings, not
the underlying idea, are what fail to transfer; other modalities simply chose
differently (VGGish for audio, ChemNet for molecules, an NB instead of a Gaussian for
counts). Strip the image-specific choices away and the ideas survive: Fréchet
distance and MMD move to PCA, an embedding, or count space; precision/recall move
unchanged as pure geometry; likelihood becomes the held-out IWAE; and per-gene
statistics were native to cells all along. The Inception Score is the one genuine
casualty, because its appeal *was* the shared classifier biology doesn't have.

*Next: back to the main spine, [Chapter 05 — Extrinsic Evaluation](05-extrinsic-evaluation.md),
where we stop asking whether the model is good on its own terms and start asking
whether its latent is good for a job — culminating in the flagship test of
predicting held-out perturbation responses.*
