# Package-Level Documentation

Technical documentation for the `genailab` package modules.

## Contents

| Document | Description |
|----------|-------------|
| [foundation/](../../foundation/README.md) | Foundation model adaptation framework |
| (Add other module docs as they are created) | |

## Subpackage Documentation

Each subpackage may have its own `docs/` directory or README:

- `foundation/` — Foundation model adaptation (LoRA, adapters, resource configs)
  - See [foundation/README.md](../../foundation/README.md)
- `data/docs/` — Data loading, transforms, batch handling
- `model/docs/` — VAE, diffusion, encoders/decoders
- `eval/docs/` — Metrics, counterfactual evaluation
- `workflows/docs/` — Training, simulation pipelines

## Documentation Guidelines

- Use Markdown format
- Include code examples where helpful
- Reference related modules with relative imports
- Keep docs close to the code they describe
