# Products

Mature, deployable applications built on genai-lab. Entries in this directory
are **things you can actually run** — stable API, versioned checkpoints,
inference docs, known limitations.

---

## Products vs Applications

| | [Applications](../applications/) | Products (here) |
|---|---|---|
| **Claim** | "Here's an approach you could use" | "Here's a thing you can run" |
| **Code location** | `examples/<topic>/` scripts | `src/genailab/applications/<name>/` |
| **API** | Whatever fits the exploration | Stable public interface |
| **Tests** | Informal | Inference path covered |
| **Artifacts** | Optional | Versioned checkpoints with metadata |
| **Baseline comparison** | Nice to have | Required |
| **Deployability** | Not expected | Inference without training infra |

Applications are a bet on methodology. Products are a commitment to users.

## Promotion Criteria

To graduate from `docs/applications/<topic>.md` to `docs/products/<name>/`,
the work must satisfy **all** of the following:

1. **Code maturity** — implementation lives under
   `src/genailab/applications/<name>/` (not `examples/`), with a stable public
   API. Signature changes constitute a breaking change.
2. **Evaluation** — benchmarked against at least one published baseline. Results
   reproducible from a clean environment with the documented command.
3. **Testability** — test suite covers the inference path at minimum (training
   path coverage is a plus but not required).
4. **Deployability** — a CLI entry point (via `pyproject.toml [project.scripts]`)
   or a library-level public function with stable signature. Inference must
   run without requiring the training-time infrastructure (e.g., no SkyPilot
   invocation inside `predict()`).
5. **Artifacts** — trained checkpoints are versioned with metadata:
   (seed, data version / hash, training config, performance on the benchmark).
   Artifacts live either on a network volume or a public host (HF Hub), not
   in git.
6. **Documentation** — `docs/products/<name>/README.md` covers:
   - What the product does and does not do
   - Installation and inference quickstart
   - Expected performance on the benchmark, with confidence intervals
   - Known limitations and failure modes
   - Accompanying runnable notebook in `notebooks/<topic>/` showing
     end-to-end use on a new input

## Current Products

_(none yet)_

The perturbation prediction flagship is currently a **Stage-2 application**
([docs/applications/perturbation_prediction.md](../applications/perturbation_prediction.md))
— methodology documented, implementation in progress. It becomes a product
once the six promotion criteria above are met.

## Demotion

Products can be demoted back to applications when they regress — e.g., an
upstream dataset changes, a baseline comparison becomes invalid, or a
dependency breaks the inference path. This is healthier than silently letting
a stale product live in the product tier.

## Adding a New Product

1. Confirm all six promotion criteria are met (not "mostly met" — all).
2. Create `docs/products/<name>/` with `README.md` following the structure
   above.
3. Remove or archive the predecessor `docs/applications/<name>.md` if it exists
   (or update it to point to the product).
4. Add a one-line entry to "Current Products" above.
5. Announce in a session summary: `dev/sessions/YYYY-MM-DD_<name>-graduated.md`.
