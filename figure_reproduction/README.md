# Result Figure Reproduction

This folder contains only the plotting source needed for the seven reference
result images distributed with the reproducibility archive.

From the release root:

```bash
uv run --locked --isolated python figure_reproduction/reproduce_all_figures.py
uv run --locked --isolated python figure_reproduction/verify_final_figures.py
```

Outputs are written to `figure_reproduction/generated/`. The verifier requires
the same canvas dimensions as the packaged manuscript images. It reports exact
pixel identity when available and otherwise permits only a tightly bounded
renderer-dependent anti-aliasing difference (mean absolute RGB error at most 5
and at most 5% changed pixels). All data-driven plots read frozen CSV files from
`06_DATA_AND_RESULTS/`; no timing experiment is silently rerun.
