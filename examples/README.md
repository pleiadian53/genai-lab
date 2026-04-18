# Examples

Production-style Python scripts demonstrating end-to-end generative AI workflows
on real biological data. Organized by topic:

```
examples/
├── perturbation/   # Perturb-seq baselines and JEPA (flagship)
├── flow_matching/  # Rectified flow experiments
└── <topic>/        # Add new topics as flagship work expands
```

## Convention

| Directory | Content | Format |
|-----------|---------|--------|
| `examples/<topic>/` | Production scripts, batch jobs, benchmarks | `.py` |
| `notebooks/<topic>/` | Tutorials, EDA, interactive exploration | `.ipynb` |
| `src/genailab/` | Reusable library code imported by both | `.py` |

Scripts here should be:
- **Runnable from the command line** — `python examples/perturbation/P1_download.py`
- **Milestone-gated** — each script produces a concrete artifact (results, metrics, model)
- **Documented via sibling `docs/`** — results and findings under `examples/<topic>/docs/`

See the existing [agentic-spliceai](https://github.com/pleiadian53/agentic-spliceai)
project for the full pattern (milestone M1-M4 layout, `ops_*_pod.sh` per task,
etc.).

---

## Running on GPU

Any script in this tree that needs a GPU should be run via the
[`ops/`](../ops/) cluster provisioner, not directly on your laptop:

```bash
# Local (laptop) — small debug runs
python examples/<topic>/<script>.py

# Remote (RunPod) — real training, benchmarking
python ops/provision_cluster.py          # spin up an A40
ssh genai-workspace
cd /workspace/genai-lab
python examples/<topic>/<script>.py
```

See [`ops/README.md`](../ops/README.md) for the full GPU workflow (data staging,
volume caching, teardown).

---

## Relationship to `notebooks/`

Examples and notebooks are **parallel views** of the same work:

- **Notebooks** for learning and visual exploration (CVAE latent-space UMAPs,
  loss curves, intermediate inspection)
- **Examples** for productionized versions of the same pipeline (headless,
  command-line, reproducible)

A well-developed topic will have both: a notebook that walks through the
intuition, and a script that runs the actual benchmark at scale.
