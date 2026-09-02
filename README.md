# Robotica reproducibility package

This is the compact MIT-licensed code and evidence package for **Time-Consistent
Risk-Sensitive Regret Synthesis for Temporal-Logic Robot Planning in Partially
Known Environments**. It is designed to verify the paper's method,
experiments, figures, and headline conclusions.

## What is included

- `02_CORE_CODE/`: scLTL automata, belief/knowledge updates, hard-feasibility
  pruning, nested risk recursion, regret oracle, and explicit/factored solvers.
- `03_EXPERIMENT_AND_BENCHMARK_CODE/`: deterministic Robotica experiment
  drivers, benchmark definitions, and core tests.
- `06_DATA_AND_RESULTS/`: frozen machine-readable evidence used by the paper.
- `01_FIGURES/`: the ten final paper figure assets.
- `figure_reproduction/`: scripts for the seven data-driven result figures.
- `claim_evidence_manifest.json` and `verify_robotica_claims.py`: the executable
  mapping from paper claims to evidence.

Figures 1-3 (`fig_problem_setting.png`, `fig_method_framework.png`, and
`fig_method_factored_solver.png`) are author-drawn method illustrations. They
are intentionally provided only as final PNG assets; this archive contains no
script that generates them. Figures 4-10 are regenerated from the included
CSV/JSON evidence.

## Reproduce

Install [uv](https://docs.astral.sh/uv/) 0.11.24 and run from this directory:

```bash
uv run --locked --isolated python reproduce.py --quick
uv run --locked --isolated python reproduce.py --full --output-root artifacts/full
uv run --locked --isolated python reproduce.py --figures --output-root artifacts/figures
uv run --locked --isolated python reproduce.py --all --output-root artifacts/all
```

`--quick` runs the core tests and 11 frozen-evidence claim checks. `--full`
recomputes the deterministic Stage 3.3 and Stage 3.5 scientific outputs and
compares them with the references. `--figures` regenerates and verifies only
the seven data-driven result figures. Generated files are written below the
chosen output directory and are not part of the release.

To audit archive contents and hashes:

```bash
uv run --locked --isolated python verify_release.py
```

See `REPRODUCIBILITY.md` for evidence levels, timing interpretation, and the
expected verification sequence. See `ROBOTICA_CLAIM_EVIDENCE.md` for the
paper-to-code map.

## Citation and license

Citation metadata are in `CITATION.cff`. The authors remain anonymous in this
review release and should replace the placeholder after de-anonymization. Code
and project-created package contents are released under the MIT License; see
`LICENSE` and `THIRD_PARTY.md`.
