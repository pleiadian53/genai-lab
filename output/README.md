# Output Directory

Derived artifacts — benchmark tables, comparison plots, paper figures, and
cross-run analyses that are **not tied to a single training run**.

## Not in Git

Only this `README.md` and the local `.gitignore` are tracked. All output
files are excluded.

## Relationship to `runs/`

`runs/` and `output/` are **both gitignored** but serve different purposes:

| Dir | Purpose | Lifetime | Example contents |
|-----|---------|----------|------------------|
| [`runs/<topic>/<run-name>/`](../runs/) | **One training run** — artifacts tied to a specific invocation | Ephemeral; may be pulled from a pod, archived, or deleted once results are extracted | `training.log`, `checkpoint_*.pt`, `model_final.pt`, `samples_final.png`, `pid`, `config.json` |
| `output/<topic>/` (here) | **Derived analyses** across runs or over time | Longer-lived; summarizes the best of what the runs produced | Benchmark tables, comparison grids, paper figures, evaluation reports, UMAPs, case studies |

A useful rule: if you could regenerate it by re-running one training script
it belongs in `runs/`. If it requires combining multiple runs, comparing
against external baselines, or rendering for a paper, it belongs in
`output/`.

## Layout

Organize by topic, matching the `examples/<topic>/` and `notebooks/<topic>/`
convention:

```
output/
├── flow_matching/
│   ├── fm_vs_ddpm_benchmark.md          # cross-method comparison
│   ├── samples_comparison.png           # epoch 50 vs 130 vs 300 grid
│   └── step_count_ablation.png          # 1 / 5 / 10 / 50 step quality
├── perturbation/
│   ├── scgen_cpa_benchmark.md           # vs published baselines
│   ├── deg_recovery.png                 # per-perturbation DEG metrics
│   └── compositional_generalization.md
└── vae/
    └── nb_vs_zinb_comparison.md
```

## When to Promote Outputs to `docs/`

Once an output is stable and presentation-ready, copy (or re-render) it into
[`docs/applications/<topic>.md`](../docs/applications/) or
[`docs/products/<name>/`](../docs/products/) as part of that topic's write-up.

`output/` is your working scratch space; `docs/` is the publishable result.

## Related

- [`runs/`](../runs/) — per-run training artifacts
- [`examples/docs/running_experiments.md`](../examples/docs/running_experiments.md) — how training runs produce `runs/<topic>/<name>/`
- [`data/README.md`](../data/README.md) — input side of the pipeline
