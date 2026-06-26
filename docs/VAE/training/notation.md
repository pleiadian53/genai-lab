# Notation Reference — Training and Evaluating VAEs

*A standalone glossary for this series, with a "read as" column. Keep it open in
a second tab while reading.*

Chapters still reintroduce the symbols they lean on — you shouldn't have to leave
the page to follow an argument — but this is the one place every symbol is
collected, so nothing is ever truly undefined. It grows as the series does; right
now it covers [Chapters 01–02](README.md).

---

## The running storyline: predicting perturbation response

Notation sticks better when it tells a story, and ours is the project's flagship:
**predict how a cell responds to a genetic perturbation** — switch a gene on and
ask what the cell does — without running every experiment at the bench. Because a
VAE is *generative*, it can even answer counterfactuals: what would *this* cell
have done under a perturbation we never tried?

We build up to that in two stages. Chapters 01–03 **warm up on PBMC** — a simpler
dataset of immune cells with known *types* and no perturbation — to learn the
training mechanics on something gentle. Chapters 04–06 **graduate to Norman 2019
Perturb-seq**, where the condition $c$ becomes the perturbation and evaluation
asks the real question: did we predict held-out responses? Here is how the
symbols map onto that story:

| Symbol | …in the perturbation story |
|---|---|
| $x$ | a cell's expression counts — its current **state** |
| $z$ | **latent cell state** — a compact "where this cell is" |
| $c$ | the **perturbation** applied (which gene or genes were switched on) |
| $q_\phi(z \mid x)$ | read a cell, infer its latent state |
| $p_\theta(x \mid z, c)$ | given a state **and a perturbation**, predict the response |
| $p(z)$ | the space of plausible cell states to sample from |

The counterfactual move the flagship rests on: **encode a control cell to $z$,
then decode under a new perturbation $c'$** to predict how that very cell would
have responded.

---

## The model

| Symbol | Read as | Meaning |
|---|---|---|
| $x$ | "x" | one **data point** — for us, one cell's vector of gene-expression counts |
| $z$ | "z" | the **latent code**: a short vector summarizing $x$ ("latent" = hidden, not directly observed) |
| $\phi$ | "phi" | the **encoder** weights |
| $\theta$ | "theta" | the **decoder** weights |
| $q_\phi(z \mid x)$ | "q-phi of z given x" | the **encoder** distribution: a learnable, cheap stand-in for the intractable true posterior |
| $p_\theta(x \mid z)$ | "p-theta of x given z" | the **decoder** distribution: how a latent is rendered back into data |
| $c$ | "c" | a **condition** fed alongside $z$ (the "C" in CVAE) — in the flagship, the **perturbation** applied to a cell |
| $p_\theta(x \mid z, c)$ | "p-theta of x given z and c" | the **conditional decoder**: predict $x$ from a latent state *and* a condition $c$ |
| $p_\theta(z \mid x)$ | — | the **true posterior** (intractable); $q_\phi$ approximates it |
| $p(z)$ | "p of z" | the **prior** over latents, almost always the standard normal $\mathcal{N}(0, I)$ |
| $\mu(x)$ | "mu" | the **mean** vector the encoder outputs for $x$ |
| $\sigma(x)$ | "sigma" | the **standard-deviation** (spread) vector the encoder outputs for $x$ |
| $\mathcal{N}(\mu, \Sigma)$ | "normal" | a **Gaussian** (bell curve) with mean $\mu$ and covariance $\Sigma$ |
| $I$ | "identity" | the identity matrix; $\mathcal{N}(0, I)$ is the standard normal |
| $\varepsilon$ | "epsilon" | fixed noise $\varepsilon \sim \mathcal{N}(0, I)$ used by the reparameterization trick |
| $\odot$ | "elementwise times" | elementwise (Hadamard) multiplication |

## The objective

| Symbol | Read as | Meaning |
|---|---|---|
| $\mathcal{L}$ | "L" | the **loss** we minimize (the negative ELBO; lower is better) |
| ELBO | — | **Evidence Lower Bound**: a tractable floor under $\log p_\theta(x)$ that we maximize |
| $\mathbb{E}_{q_\phi(z \mid x)}[\cdot]$ | "expectation" | an **average** over latent codes $z$ drawn from the encoder; estimated by sampling |
| $\log p_\theta(x \mid z)$ | — | the decoder's **log-likelihood** of the real $x$ — bigger means a more faithful reconstruction |
| $\text{KL}(q \Vert p)$ | "KL divergence" | a number measuring how far $q$ is from $p$; zero when they match, growing as they diverge |
| $\beta$ | "beta" | a knob reweighting the KL term ($\beta = 1$ is the standard VAE; introduced in [Chapter 03](03-the-training-loop.md)) |

## The data

| Symbol | Read as | Meaning |
|---|---|---|
| $X$ | "X" | the whole dataset as a matrix |
| $X \in \mathbb{R}^{N \times D}$ | — | a table of real numbers with $N$ rows and $D$ columns |
| $N$ | "N" | the number of **samples** (rows) — cells, for us |
| $D$ | "D" | the number of **features** (columns) — genes, for us |
| $X_{ij}$ | — | the entry in row $i$, column $j$ — the count of gene $j$ in cell $i$ |
| $L_i$ | "L-i" | the **library size** of cell $i$: its total counts, a mostly technical depth measure |
| $s$ | "s" | a fixed **target sum** used when normalizing (a scale constant) |
| log1p | "log-one-p" | the map $u \mapsto \log(1 + u)$, a log that is safe at zero |
| HVG | — | **highly variable genes**: the most informative genes, kept after selection |
| dropout | — | a technical **false zero** — a present molecule the measurement missed |
| NB / ZINB | — | **Negative-Binomial** / **Zero-Inflated NB**: count distributions used by the decoder |

## Richer posteriors ([Chapter 01a](01a-richer-posteriors.md))

| Symbol | Read as | Meaning |
|---|---|---|
| $f_1, \ldots, f_K$ | — | a chain of learned **invertible** transforms in a normalizing flow |
| $z_0, z_K$ | — | the simple base sample and the flow's complex-shaped output |
| $\left\lvert \det \frac{\partial f_k}{\partial z_{k-1}} \right\rvert$ | "Jacobian determinant" | how much transform $f_k$ locally expands or shrinks volume |
| $L$ | "L (Cholesky)" | the Cholesky factor of a full covariance, used in $z = \mu + L\varepsilon$ |

---

*The evaluation symbols (reconstruction error, active units, ARI, NMI, and the
rest) join this table as [Chapters 04–05](README.md) introduce them.*
